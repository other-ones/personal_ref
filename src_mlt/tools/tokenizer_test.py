from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor

tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="tokenizer",
        revision=None,
    )

caption='words'
word_token_ids=tokenizer.encode(caption,add_special_tokens=False)
print(word_token_ids,'token_ids')
decoded=tokenizer.decode(word_token_ids)
print(decoded,'decoded')