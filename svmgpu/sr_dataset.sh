#!/bin/bash
# for i in {1..1000}
# do
# 	foo=$(printf "%04d" $i)
# 	mkdir ./dataset/train_1_5/$foo
# done

# index=2667
# for f in /home/lamluong/Documents/module/detect_mask/test_model/data/nomask/*.jpg
# do
# #  echo $f;
#    cp $f /home/lamluong/Documents/module/detect_mask/test_model/data/nomask_backup/$index.jpg
# #  mv /home/lamluong/Documents/module/detect_mask/test_model/data/nomask/$index.jpg
#    index=$((index+1))
# #  mv "$f" "1-$((i++))-3.jpg";
# done
index=1
for i in {1..1000}
do
	foo=$(printf "%04d" $i)
	cp /home/lamluong/Documents/module/detect_mask/test_model/data/nomask_backup/$index.jpg ./dataset/train_1_1/$foo
  index=$((index+1))
  # cp /home/lamluong/Documents/module/detect_mask/test_model/data/nomask_backup/$index.jpg ./dataset/train_1_5/$foo
  # index=$((index+1))
  # cp /home/lamluong/Documents/module/detect_mask/test_model/data/nomask_backup/$index.jpg ./dataset/train_1_5/$foo
  # index=$((index+1))
  # cp /home/lamluong/Documents/module/detect_mask/test_model/data/nomask_backup/$index.jpg ./dataset/train_1_5/$foo
  # index=$((index+1))
  # cp /home/lamluong/Documents/module/detect_mask/test_model/data/nomask_backup/$index.jpg ./dataset/train_1_5/$foo
  # index=$((index+1))
done
