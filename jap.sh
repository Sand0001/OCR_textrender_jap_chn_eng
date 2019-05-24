#python main.py --num_img 200000
source activate ocr
rm -rf output/default
rm -rf test/default
tag="japeng"
python main.py --num_img 5800000 --tag ${tag}  --output_dir output
python main.py --num_img 40000 --tag ${tag} --output_dir test
cd /home/denghailong/ocr/
source activate ocr
python dl_resnet_crnn.py
