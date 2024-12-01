import organ
from organ import ORGAN

model = ORGAN('test', 'mol_metrics', params={'PRETRAIN_DIS_EPOCHS': 1, 'PRETRAIN_GEN_EPOCHS':100})
model.load_training_set('./data/qm9_5k.csv')
# model.set_training_program(['novelty'], [1])
model.set_training_program(['druglikeliness'], [100])
model.load_metrics()
# model.load_prev_training(ckpt='./ckpt/test_pretrain_ckpt')
model.organ_train(ckpt_dir='ckpt')
# 调用训练好的模型
# model.generate(100, label_input=False)
# # 使用条件训练
# model.conditional_train(ckpt_dir='ckpt')

# # 生成特定类别的分子
# molecules = model.generate_samples(100, label_input=True, target_class=0)  # 生成第0类分子