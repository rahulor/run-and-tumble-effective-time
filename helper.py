import pandas as pd
import pickle

def pickle_to(filename, data):
    pikd = open(filename + '.pickle', 'wb')
    pickle.dump(data, pikd)
    pikd.close()
    
def unpickle_from(filename):
    pikd = open(filename + '.pickle', 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data

def write_dataframe(df, fname):
    pathcsv = 'data/' + fname + '.csv'
    pathtxt = 'data/' + fname + '.txt'
    df.to_csv(pathcsv, index=False)
    file = open(pathtxt, "w")
    text = df.to_string()
    #file.write("RESULT\n\n")
    file.write(text)
    file.close()

def sort_data():
    result_all = []
    LogList = unpickle_from('data/log')
    for log in LogList:
        row = {}
        row.update(log.args)
        print(row)
        m = log.df_sim.mean(axis = 0)
        s = log.df_sim.sem(axis = 0)
        m.index = [name+'_mean' for name in m.index]
        s.index = [name+'_sem' for name in s.index]
        ms = {**m, **s} 
        row.update(ms)
        result_all.append(row)
    df = pd.DataFrame(result_all)
    write_dataframe(df, 'result_all')
if __name__ == '__main__':
    pass
    