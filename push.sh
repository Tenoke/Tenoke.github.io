JEKYLL_ENV=production jekyll build
git add .
git commit -m "$1"
git push
cp -r _site/* ../Tenoke.github.io/
cd ../Tenoke.github.io/
git add .
git commit -m "$1"
git push 
cd ../jekyll-blog