from bfgs import relax
from random_sampler import Sampler
import knowledge
import wandb
import numpy as np

wandb.login()
wandb.init(project="Baseline", entity="kly20")

MAX_RELAX_STEPS=10000
TRIES=1000
global_minimal=knowledge.global_minimal

wandb.config={
    "max_relax_steps":MAX_RELAX_STEPS,
    "tries":TRIES,
}

if __name__=="__main__":
    for num_atoms in range(60,151):
        target_energy=global_minimal[num_atoms.__str__()]
        sampler=Sampler(num_atoms,2.5,1)
        sum_step=0
        times=0
        energies=[]
        steps=[]
        for i in range(TRIES):
            comform=sampler.single_sample()
            step,energy=relax(comform.tolist(),MAX_RELAX_STEPS)
            energies.append(energy)
            steps.append(step)
            if np.abs(energy-target_energy)<=1e-2:
                sum_step+=step
                times+=1
        average_step=sum_step/times
        print("num_atoms:",num_atoms,"   average_step:",average_step)
        data = [[x, y] for (x, y) in zip(energies, steps)]
        table = wandb.Table(data=data, columns = ["minimal", "relax steps"])
        wandb.log({"Num_atoms: "+num_atoms.__str__() : wandb.plot.scatter(table, "minimal", "relax steps")})
