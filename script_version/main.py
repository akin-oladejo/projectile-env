from projectile import Projectile
from q_learning import Q


sim = Projectile() # instantiate projectile env object

model = Q(sim, exploration_fraction=0.5) # instantiate Q object

model.train() # train q-learning model

model.show()