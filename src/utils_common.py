import pandas as pd

class CustomInitializer:
    def __call__(self, shape, dtype=None):
#        return b_init(shape, dtype=dtype)
        return b_init(shape)


def _im_cols(df):
    ''' select image embedding columns in dataframe '''
    im_cols = []
    for c in df.columns:
        if c[:2] == 'im':
            im_cols += [c]
    return im_cols 


def recordSetting(class_,save_dir):
    ''' Save class attributes as setting.txt '''
    temp = []
    with open(save_dir + '/setting.txt','a') as f:
        for k,v in class_.__dict__.items():
            if str(k) == 'X_train' or str(k) == 'X_test' or str(k) == 'X_valid' or str(k) == 'y_train' or str(k) == 'y_test' or str(k) == 'df':
                continue
            elif str(k) == 'X_cols':
                f.write(str(k) + '\t\t' + str(v[len(v)-5:]) + '\n')
            elif 'HBresults_' in str(k):
                f.write(str(k) + '\t\t' + str(v) + '\n')                        
            elif 'HB' in str(k):
                continue
            elif str(k) == 'get_params':
                try:
                    f.write(str(k) + '\t\t' + json.dumps(v) + '\n')
                except:
                    pass
            else:
                f.write(str(k) + '\t\t' + str(v) + '\n')
        f.write('##################################\n\n')


def matchRate(df,correctTable='logs/UF3808_20200428/correction_0428_0124.csv'):
    ''' Calculate how many test images match with assigned class '''
    dic1 = {}
    try:
        for i,row in pd.read_csv(correctTable).iterrows():
            dic1[row['class_in_test']] = row['class_in_train'].split(',')
    except ValueError:
        for i,row in correctTable.iterrows():
            dic1[row['class_in_test']] = row['class_in_train'].split(',')

    df['match'] = df.apply(lambda x: 1 if x['pred_set'] in dic1[x['set']] else 0,axis=1)
    df['matchRate'] = [sum(df['match'])/len(df)]*len(df)
    df1 = pd.DataFrame()
    dic1 ={}
    for i,g in df.groupby('set'):
        g['MR_by_set'] = [sum(g['match'])/len(g)]*len(g)
        vc = g['pred_set'].value_counts()
        g['most_pred_set'] = [vc.index[0]]*len(g)
        g['pred_share'] = [vc.iloc[0]/len(g)]*len(g)
        dic1[max(g['set'])] = {'MR_by_set':max(g['MR_by_set']),'most_pred_set':max(g['most_pred_set']),'pred_share':max(g['pred_share'])}
        df1 = pd.concat([df1,g],axis=0,ignore_index=True)
    return df1
