import pandas as pd
import glob
import shutil

filename = "./model/Analyse_Ray_log3.xlsx"
sheetname = "ray11"
exp_path = "../ray_results/"
exp_name = "objective_2023-04-21_17-36-40"
path_out = "./model/best/"

db = pd.read_excel(filename, sheet_name=sheetname)

# get the 10 best recorded
best_recorded = db[db.loc[:,"dropout"]>0.1].nsmallest(10,"diff_loss").reset_index()

# go through the best recorded and copy the checkpoint.pt file to a new folder
names = []
for count,i in enumerate(best_recorded.loc[:,"Trial name"]):
    folder = glob.glob(exp_path+exp_name+i.strip()+"*")[0]
    check_path = exp_path+folder+"/checkpoint_000000/checkpoint.pt"
    
    nb_layers = best_recorded.loc[count,"nb_layers"]
    nb_neurons = best_recorded.loc[count,"nb_neurons"]
    p_drop = best_recorded.loc[count,"dropout"]
    
    name_out = "l"+str(nb_layers)+"_n"+str(nb_neurons)+"_p"+str(p_drop)+"_mGELU_cpfree"+".pth"
    
    shutil.copyfile(check_path, path_out+name_out)
    
    names.append(name_out)
    
best_list = pd.DataFrame(names, columns=["name"])
best_list.to_csv(path_out+"best_list.csv")