import cv2
import pdb
import itertools
from typing import Any, Callable, Dict, Optional, Union, List

# import spacy
import torch
import sys
sys.path.insert(0, './packages')
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    EXAMPLE_DOC_STRING,
    rescale_noise_cfg
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteAttnProcessor
)
import numpy as np
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from compute_loss import get_attention_map_index_to_wordpiece, split_indices, calculate_positive_loss, calculate_negative_loss, get_indices, start_token, end_token, align_wordpieces_indices, extract_attribution_indices, extract_attribution_indices_with_verbs, extract_attribution_indices_with_verb_root, extract_entities_only,_symmetric_kl



logger = logging.get_logger(__name__)


class StableDiffusionPipelineKeywordCFGSchedule(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 requires_safety_checker: bool = True,
                 include_entities: bool = False,
                 ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)

        # self.parser = spacy.load("en_core_web_trf")
        self.subtrees_indices = None
        self.doc = None
        self.include_entities = include_entities
        
    
    def _aggregate_and_get_attention_maps_per_token(self,batch_size,do_classifier_free_guidance):
        # FIXED
        # aggregate_attention: get all the attn maps of target resolution and average them
        # attention_maps_batch: (bsz,16,16,77) list
        attention_maps_batch = self.attention_store.aggregate_attention( #defined at pipline.py
            from_where=("up", "down", "mid"),
            batch_size=batch_size,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        # print(torch.max(attention_maps),torch.min(attention_maps),'attention_maps')
        # attention_maps_list: [16,16,77] -> [77,16,16] reordering
        attention_maps_batch = _get_attention_maps_list(
            attention_maps_batch=attention_maps_batch
        )
        # for item in attention_maps_batch:
        #     print(item.shape,'attn_map.shape')
        # attention_maps_batch:bsz,77,16,16
        return attention_maps_batch

    @staticmethod
    def _update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        torch.autograd.set_detect_anomaly(True)
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            # attention_store: AttentionStore 
            # (this is deinfed at pipeline_stable_diffusion_attend_and_excite.py)
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )
            # print(name,'register_attention_control AttendExciteAttnProcessor')

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt_batch: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            attn_res=None,
            parsed_prompt: str = None,
            render_embs=None,
            mask_tensors=None,
            attn_mod_params=None, #[bsz,77]
            verbose=True, #[bsz,77]
            g_schedule1: [float]=None,
            g_schedule2: [float]=None,
            g_schedule3: [float]=None,
    ):
        r"""
        The call function to the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            attn_res (`tuple`, *optional*, default computed from width and height):
                The 2D resolution of the semantic attention map.
            syngen_step_size (`float`, *optional*, default to 20.0):
                Controls the step size of each SynGen update.
            num_intervention_steps ('int', *optional*, defaults to 25):
                The number of times we apply SynGen.
            parsed_prompt (`str`, *optional*, default to None).


        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        # if parsed_prompt:
        #     self.doc_list = parsed_prompt
        # else:
        #     self.doc_list = {prompt:self.parser(prompt) for prompt in prompt_batch}
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt_batch,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt_batch is not None and isinstance(prompt_batch, str):
            batch_size = 1
        elif prompt_batch is not None and isinstance(prompt_batch, list):
            batch_size = len(prompt_batch)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt_batch,
            self.unet.device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        

        # 4. Prepare timesteps
        # print(num_inference_steps,'num_inference_steps')
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attn_res = attn_res
        # print('AttentionStore initialize')
        self.attention_store = AttentionStore(self.attn_res)
        # every forward setp AttentionStore stores attention values of target resolution
        # only cross attn is considered
        # (this is deinfed at pipeline_stable_diffusion_attend_and_excite.py)


        # AttentionStore is register to attention processor here
        # everytime the attention is called, AttendExciteAttnProcessor calls AttentionStore to store them
        self.register_attention_control() 
        # replace the attention computation module with AttendExciteAttnProcessor: 
        # every forward pass, this instances are called and 
        # actual computation of attention is processed in AttendExciteAttnProcessor

        
        # 7. Denoising loop #HERE
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        mask_tensors=mask_tensors.to(self.unet.device)
        render_embs=render_embs.to(self.unet.device)
        text_embeddings = (
            prompt_embeds[batch_size:] if do_classifier_free_guidance else prompt_embeds #cond_input
        )
        
        with torch.no_grad():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for tidx, tstep in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = (torch.cat([latents] * 4) if do_classifier_free_guidance else latents)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, tstep)
                    
                    # print(latent_model_input.shape,'latent_model_input.shap1
                    if do_classifier_free_guidance:
                        # # # # # # # 
                        # Latent Repeating
                        # 0:4 latent
                        # 4:5 mask
                        # 5:9 render
                        latent_model_input = torch.cat([latent_model_input, 
                                                        mask_tensors.repeat(4*num_images_per_prompt,1,1,1), 
                                                        render_embs.repeat(4*num_images_per_prompt,1,1,1)], dim=1)
                        # uncond/ref/text/both
                        # 1) uncond - zero mask&render
                        latent_model_input[:(num_images_per_prompt*batch_size), 4:, :, :] *= 0.0 #uncond/ref/text/all
                        # 2) ref - all alive - No masking
                        # 3) text - zero mask/render
                        latent_model_input[(num_images_per_prompt*batch_size*2):(num_images_per_prompt*batch_size*3), 4:, :, :] *= 0.0 #uncond/ref/text
                        # 4) all - all alive - No masking
                        # Latent Repeating
                        # # # # # # # 
                    else:
                        latent_model_input = torch.cat([latent_model_input, mask_tensors, render_embs],dim=1)

                    # predict the noise residual
                        # treg=torch.pow(tstep/1000,5)*guidance_strength
                    # treg=0.4 
                    # treg=1
                    # treg=treg*guidance_strength
                    # attn_mod_params["treg"]=treg
                    noise_pred = self.unet(
                        latent_model_input,
                        tstep,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                        attn_mod_params=attn_mod_params,
                    )[0]

                    
                    # perform guidance
                    if do_classifier_free_guidance:
                        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(4)
                        # noise_pred = noise_pred_uncond + guidance_scale * (
                        #         noise_pred_text - noise_pred_uncond
                        # )
                        noise_pred_uncond, noise_pred_ref,noise_pred_text,noise_pred_both = noise_pred.chunk(4)
                        noise_pred = noise_pred_uncond + g_schedule1[tidx] *(noise_pred_ref-noise_pred_uncond) + g_schedule2[tidx]*( noise_pred_text - noise_pred_uncond) \
                                    + g_schedule3[tidx] * (noise_pred_both - noise_pred_uncond)


                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, tstep, latents, **extra_step_kwargs, return_dict=False)[0]
                    
                    # call the callback, if provided
                    if tidx == len(timesteps) - 1 or (
                            (tidx + 1) > num_warmup_steps and (tidx + 1) % self.scheduler.order == 0
                    ):
                        if verbose:
                            progress_bar.update()
                        if callback is not None and tidx % callback_steps == 0:
                            callback(tidx, tstep, latents)

            # print('Retrieve attention maps')
            attention_maps = self._aggregate_and_get_attention_maps_per_token(batch_size=batch_size,
            do_classifier_free_guidance=do_classifier_free_guidance)
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept),attention_maps
    

                
    def _get_attn_map(
        self,
        batch_size,
        token_lengths_batch,
    ):

        attn_maps_batch=[]
        for token_lengths in token_lengths_batch:
            attn_maps=[]
            #[0,100]
            attention_maps=torch.stack(attention_maps)
            
            for aidx,item in enumerate(attention_maps):
                if aidx>=token_lengths+1:
                    break
                item=item.detach().cpu().numpy()
                attn_maps.append(item)
            attn_maps_batch.append(attn_maps)
    

    


def _get_attention_maps_list(
        attention_maps_batch: torch.Tensor
) -> List[torch.Tensor]:
    # attention_maps_batch: [bsz, 16, 16, 77] 
    # returns -> [bsz,77,16,16]
    # attention_maps *= 100
    # FIXED
    attention_maps_list_batch=[]
    for attention_maps in attention_maps_batch:
        # attention_maps: [16,16,77] tensor
        attention_maps_list = [attention_maps[:, :, i] for i in range(attention_maps.shape[2])] #16,16,77 -> 77,16,16
        attention_maps_list_batch.append(attention_maps_list)
    # attention_maps_list_batch:[bsz,77,16,16]
    return attention_maps_list_batch


def unify_lists(list_of_lists):
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list):
                yield from flatten(elem)
            else:
                yield elem

    def have_common_element(lst1, lst2):
        flat_list1 = set(flatten(lst1))
        flat_list2 = set(flatten(lst2))
        return not flat_list1.isdisjoint(flat_list2)

    lst = []
    for l in list_of_lists:
        lst += l
    changed = True
    while changed:
        changed = False
        merged_list = []
        while lst:
            first = lst.pop(0)
            was_merged = False
            for index, other in enumerate(lst):
                if have_common_element(first, other):
                    # If we merge, we should flatten the other list but not first
                    new_merged = first + [item for item in other if item not in first]
                    lst[index] = new_merged
                    changed = True
                    was_merged = True
                    break
            if not was_merged:
                merged_list.append(first)
        lst = merged_list

    return lst
