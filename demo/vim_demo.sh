GPU=2 \

export CUDA_VISIBLE_DEVICES=${GPU}

#
# Generate image from fMRI
#
python demo/vim2_demo.py \
--cfg demo/cfg/vim2-demo.yml \
--gpu ${GPU}
