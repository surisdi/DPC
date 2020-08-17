python main.py \
--gpu 0,1,2,3 \
--net resnet18 \
--dataset k600 \
--batch_size 64 \
--img_dim 80 \
--epochs 300 \
--start-epoch 0 \
--reset_lr \
--hyperbolic \
--hyperbolic_version 1 \
--distance 'regular' \
--lr 1e-5 \
--prefix hyperbolic_v1_regular_lr5
