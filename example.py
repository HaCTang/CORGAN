import organ
from organ import ORGAN

model = ORGAN('test', 'mol_metrics', params={'PRETRAIN_DIS_EPOCHS': 1, 'PRETRAIN_GEN_EPOCHS':100})
model.load_training_set('./data/qm9_5k.csv')
# model.set_training_program(['novelty'], [1])
model.set_training_program(['druglikeliness'], [100])
model.load_metrics()
# model.load_prev_training(ckpt='./ckpt/test_pretrain_ckpt')
model.train(ckpt_dir='ckpt')
