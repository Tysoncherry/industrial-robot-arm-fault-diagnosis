from simulation.robot_model import MobileRobot
import pandas as pd, joblib, os
from extract_features import WINDOW, RAW_DIR, OUT, extract_features_from_window

model = joblib.load('ml_model/best_svm_model.joblib')
scaler = joblib.load('ml_model/scaler.joblib')

fault='voltage_drop'
robot=MobileRobot(); robot.set_fault(fault)
rows=[]
for _ in range(int(10/robot.dt)):
    data=robot.update(6,6)
    rows.append([data[k] for k in ['time','x','y','theta','encoder_left','encoder_right','imu_gyro','imu_ax','imu_ay']]+[fault])
raw=pd.DataFrame(rows,columns=['time','x','y','theta','encoder_left','encoder_right','imu_gyro','imu_ax','imu_ay','label'])
raw.to_csv('results/integration_raw.csv',index=False)

# feature
feats=[]
df=raw
for i in range(0,len(df)-WINDOW+1,WINDOW):
    w=df.iloc[i:i+WINDOW]
    feats.append(extract_features_from_window(w))
pf=pd.DataFrame(feats)
pf['label']=fault
pf.to_csv('results/integration_features.csv',index=False)

# predict
X=scaler.transform(pf.drop(columns=['label']))
pf['pred']=model.predict(X)
pf.to_csv('results/integration_predictions.csv',index=False)
print('Integration done.')