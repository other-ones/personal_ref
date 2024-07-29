current_dir=$(pwd)
cd ../;
git add .;
git commit -m "update";
git push origin main;
cd "$current_dir"