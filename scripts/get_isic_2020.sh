mkdir -p data && cd data
kaggle datasets download -d cdeotte/melanoma-512x512
mkdir isic2020
mv melanoma-512x512.zip isic2020/melanoma-512x512.zip
cd isic2020
unzip melanoma-512x512.zip
rm melanoma-512x512.zip
