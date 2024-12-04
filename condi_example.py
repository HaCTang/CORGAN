import organ
from organ import ORGAN

model = ORGAN('cond_test', 'mol_metrics', params={
    'CLASS_NUM': 2,
    'PRETRAIN_DIS_EPOCHS': 1,
    'PRETRAIN_GEN_EPOCHS': 250,
    'MAX_LENGTH': 100
})

# 加载训练数据和设置训练程序
model.load_training_set('./data/train_NAPro.csv')
model.set_training_program(['druglikeliness'], [5])
model.load_metrics()

# model.organ_train(ckpt_dir='ckpt')
# 使用条件训练
model.load_prev_training(ckpt='./checkpoints/cond_test/cond_test_4.ckpt')
model.conditional_train(ckpt_dir='ckpt', gen_steps=2)

# 生成特定类别的分子
molecules = model.generate_samples(100, label_input=True, target_class=0)  # 生成第0类分子