#python main.py --num_img 200000
source activate ocr
rm -rf output/default
rm -rf test/default
tag="multilan_final"
python main.py --num_img 2500000 --tag ${tag}  --output_dir output
python main.py --num_img 20000 --tag ${tag} --output_dir test
#source activate ocr
#python dl_crnn.py
