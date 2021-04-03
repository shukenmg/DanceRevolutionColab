#! /bin/bash
version="layers2_win100_schedule100_condition10_detach"
e="3000"
random="_random"

# Test
python3 test.py --train_dir ../data/train_1min --test_dir ../data/test_1min \
--output_dir outputs/${version}/epoch_${e}${random} --model checkpoints/${version}/epoch_${e}.pt \
--visualize_dir visualizations/${version}/epoch_${e}${random}

files=$(ls visualizations/${version}/epoch_${e}${random})
for filename in $files
do
	ffmpeg -r 15 -i visualizations/${version}/epoch_${e}${random}/${filename}/frame%06d.jpg -vb 20M -vcodec mpeg4 \
	 -y visualizations/${version}/epoch_${e}${random}/${filename}.mp4
	echo "make video ${filename}"
done

