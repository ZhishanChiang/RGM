pwd
echo '----start train----'
echo 'Partial2Partial_exp'
python train.py --cfg experiments/train_RGM_Seen_Crop_modelnet40_transformer.yaml
# echo 'Unseen_Partial2Partial_exp'
# python train.py --cfg experiments/train_RGM_Unseen_Crop_modelnet40_transformer.yaml
# echo '=== ablation study exp ==='
# echo 'NO TRANSFORMER HAVE AIS'
# python train.py --cfg experiments/train_RGM_Unseen_Crop_modelnet40_NoAttention.yaml
# python train.py --cfg experiments/train_RGM_Unseen_Crop_modelnet40_NoAttention_nn.yaml
# echo 'NO AIS HAVE TRANSFORMER'
# python train.py --cfg experiments/train_RGM_Unseen_Crop_modelnet40_NoAIS.yaml