current_path=$(pwd)
cd ../;
git add .;
git commit -m "regular commit";
git push origin main;
cd "$current_path"
