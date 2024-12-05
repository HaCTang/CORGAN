import organ
from organ import ORGAN

model = ORGAN('cond_test', 'mol_metrics', params={
    'CLASS_NUM': 2,
    'PRETRAIN_DIS_EPOCHS': 1,
    'PRETRAIN_GEN_EPOCHS': 250,
    'MAX_LENGTH': 100
})

model.load_training_set('./data/train_NAPro.csv')
model.set_training_program(['druglikeliness'], [10])
model.load_metrics()

# model.organ_train(ckpt_dir='ckpt')
# model.load_prev_training(ckpt='./checkpoints/cond_test/cond_test_2.ckpt')
# model.conditional_train(ckpt_dir='ckpt', gen_steps=4)

model.load_prev_training(ckpt='ckpt/cond_test_class0_final.ckpt')

# 然后生成样本
model.output_samples(1000, label_input=True, target_class=0, 
                     class_generator=model.class_generators[0])
