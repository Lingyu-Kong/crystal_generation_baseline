import knowledge
import numpy as np
import wandb

from bfgs import relax
from random_sampler import Sampler

wandb.login()
wandb.init(project="go-explore", entity="kly20",name="baseline_go_explore")

MAX_RELAX_STEPS=10000
TRIES=1100
NUM_ATOMS=40
global_minimal=knowledge.global_minimal
target_energy=global_minimal[NUM_ATOMS.__str__()]

if __name__=="__main__":
    sampler=Sampler(NUM_ATOMS,2,1)
    conforms=np.zeros((TRIES,NUM_ATOMS,3))
    retry=True
    conforms_top=0
    min_energy=0
    path=[]
    while conforms_top<TRIES:
        conform=sampler.single_sample()
        step,energy,conform=relax(conform.tolist(),MAX_RELAX_STEPS)
        if np.abs(energy-target_energy)>=1e-2:
            conforms[conforms_top,:,:]=conform
            conforms_top+=1
        print("conforms_top:",conforms_top)
        if energy<min_energy:
            min_energy=energy
        path.append(min_energy)
        wandb.log({"min_energy":min_energy,})
    conforms=conforms.reshape(TRIES,NUM_ATOMS*3)
    np.savetxt(NUM_ATOMS.__str__()+".csv",conforms,delimiter=",")
    print("path: ",path)
    path=np.array(path)
    np.savetxt(NUM_ATOMS.__str__()+"_path.csv",path,delimiter=",")
            