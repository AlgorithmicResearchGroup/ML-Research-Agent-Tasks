import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from PIL import Image
import os
from torchvision import transforms
from torchvision.models.mobilenetv2 import mobilenet_v2
import random
from sklearn import metrics
import torch.optim as optim
from tqdm import tqdm
from glob import glob
import transformers
from sklearn.model_selection import StratifiedKFold

import torch
import subprocess
import argparse
import threading
import time

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform(image_size,mean,std):
    return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)])

class UMNIST(Dataset):    
    def __init__(self,df,img_dir,transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        image_id = self.df.iloc[index]['image_id']
        label = int(self.df.iloc[index]['digit_sum'])
        path = os.path.join(self.img_dir,f'{image_id}.jpeg')
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label)
    
class UMNIST_test(Dataset):    
    def __init__(self,img_paths,transform=None):
        self.img_paths = img_paths
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self,index):
        image_id = self.img_paths[index].split('/')[-1].split('.')[0]
        path = self.img_paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image,image_id

class Trainer():
    def __init__(self,
                train_dataset,
                val_dataset,
                test_dataset,
                train_batch_size,
                infer_batch_size,
                num_workers,
                device,
                epochs,
                model,
                optimizer,
                scheduler,
                model_save_dir,
                model_load_dir,
                test_output_dir):
        
        self.epochs = epochs
        self.epoch = 0
        self.accelarator = device
        self.model = model
        self.model_save_dir = model_save_dir
        self.model_load_dir = model_load_dir
        self.test_output_dir = test_output_dir

        self.train_dataset = train_dataset 
        self.val_dataset = val_dataset 
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset,batch_size=train_batch_size,shuffle=True,num_workers=num_workers)
        if self.val_dataset is not None:
            self.validation_loader = DataLoader(self.val_dataset,batch_size=infer_batch_size,shuffle=False,num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset,batch_size=infer_batch_size,shuffle=False,num_workers=num_workers)

        print(f"Length of train loader: {len(self.train_loader)}, {f'validation loader:{len(self.validation_loader)}'  if self.val_dataset is not None else ''}, test loader: {len(self.test_loader)}")
    
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def get_metrics(self,predictions,actual,isTensor=False):
        if isTensor:
            p = predictions.detach().cpu().numpy()
            a = actual.detach().cpu().numpy()
        else:
            p = predictions
            a = actual
        accuracy = metrics.accuracy_score(y_pred=p,y_true=a)
        return {
            "accuracy": accuracy
        }

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train_step(self):
        self.model.train()
        print("Train Loop!")
        running_loss_train = 0.0
        num_train = 0
        train_predictions = np.array([])
        train_labels = np.array([])
        for images,labels in tqdm(self.train_loader):
            images = images.to(self.accelarator)
            labels = labels.to(self.accelarator)
            num_train += labels.shape[0]
            self.optimizer.zero_grad()
            outputs = self.model(images)
            _,preds = torch.max(outputs,1)
            train_predictions = np.concatenate((train_predictions,preds.detach().cpu().numpy()))
            train_labels = np.concatenate((train_labels,labels.detach().cpu().numpy()))

            loss = self.criterion(outputs,labels)
            running_loss_train += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.lr = self.get_lr(self.optimizer)

        train_metrics = self.get_metrics(train_predictions,train_labels)
        print(f"Train Loss: {running_loss_train/num_train}")
        print(f"Train Accuracy Metric: {train_metrics['accuracy']}")    
        return {
                'loss': running_loss_train/num_train,
                'accuracy': train_metrics['accuracy'],
            }


    def val_step(self):
        if self.val_dataset is None:
            return 
        val_predictions = np.array([])
        val_labels = np.array([])
        running_loss_val = 0.0
        num_val = 0
        self.model.eval()
        with torch.no_grad():
            print("Validation Loop!")
            for images,labels in tqdm(self.validation_loader):
                images = images.to(self.accelarator)
                labels = labels.to(self.accelarator)
                outputs = self.model(images)
                num_val += labels.shape[0]
                _,preds = torch.max(outputs,1)
                val_predictions = np.concatenate((val_predictions,preds.detach().cpu().numpy()))
                val_labels = np.concatenate((val_labels,labels.detach().cpu().numpy()))


                loss = self.criterion(outputs,labels)
                running_loss_val += loss.item()
            val_metrics = self.get_metrics(val_predictions,val_labels)
            print(f"Validation Loss: {running_loss_val/num_val}")
            print(f"Val Accuracy Metric: {val_metrics['accuracy']} ")    
            return {
                'loss': running_loss_val/num_val,
                'accuracy': val_metrics['accuracy'],
            }
    def test_step(self):
        if self.model_load_dir != '':
            checkpoint = torch.load(
                f"{self.model_load_dir}/best_val_accuracy.pt")
            start_epoch = checkpoint['epoch']
            print(
                f"Model already trained for {start_epoch} epochs.")
            print(self.model.load_state_dict(checkpoint['model1_weights']))
            
        test_image_ids = np.array([])
        test_predictions = np.array([])
        self.model.eval()
        with torch.no_grad():
            print("Test Loop!")
            for images,image_ids in tqdm(self.test_loader):
                images = images.to(self.accelarator)
                outputs = self.model(images)
                _,preds = torch.max(outputs,1)
                test_image_ids = np.concatenate((test_image_ids,image_ids))
                test_predictions = np.concatenate((test_predictions,preds.detach().cpu().numpy()))
            
            df = pd.DataFrame({
                'image_id':test_image_ids,
                'digit_sum':test_predictions
            })
            df['digit_sum'] = df['digit_sum'].astype(int)
            df.to_csv(os.path.join(self.test_output_dir , 'submission.csv'),index=False)

    def run(self,run_test=True):
        best_validation_loss = float('inf')
        best_validation_accuracy = 0

        for epoch in range(self.epochs):
            print("="*31)
            print(f"{'-'*10} Epoch {epoch+1}/{self.epochs} {'-'*10}")
            train_logs = self.train_step()
            self.epoch = epoch
            
            if self.val_dataset is not None:
                val_logs = self.val_step() 
                if val_logs["loss"] < best_validation_loss:
                    best_validation_loss = val_logs["loss"]
                    torch.save({
                        'model1_weights': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'epoch': epoch+1,
                    }, f"{self.model_save_dir}/best_val_loss.pt")
                if val_logs['accuracy'] > best_validation_accuracy:
                    best_validation_accuracy = val_logs['accuracy']
                    torch.save({
                        'model1_weights': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'epoch': epoch+1,
                    }, f"{self.model_save_dir}/best_val_accuracy.pt")
        if run_test:
            self.test_step()

        return {
            'best_accuracy':best_validation_accuracy,
            'best_loss': best_validation_loss,
        }    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for training baseline on ImageNet100")
    parser.add_argument('--root_dir',default='./',type=str)
    parser.add_argument('--epochs',default=100,type=int)
    parser.add_argument('--train_batch_size',default=8,type=int)
    parser.add_argument('--infer_batch_size',default=48,type=int)
    parser.add_argument('--image_size',default=1024,type=int)
    parser.add_argument('--seed',default=42,type=int)
    parser.add_argument('--num_classes',default=28,type=int)
    parser.add_argument('--num_workers',default=2,type=int)
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--output_dir',default='./',type=str)
    parser.add_argument('--model_save_dir', default='./models', type=str)
    parser.add_argument('--model_load_dir',default='',type=str)
    args = parser.parse_args()
    EPOCHS = args.epochs
    ROOT_DIR = args.root_dir
    IMAGE_SIZE = args.image_size
    EPOCHS = args.epochs
    SEED = args.seed
    WARMUP_EPOCHS = 2
    TRAIN_BATCH_SIZE = args.train_batch_size
    INFER_BATCH_SIZE = args.infer_batch_size
    NUM_CLASSES = args.num_classes
    NUM_WORKERS =args.num_workers
    LEARNING_RATE = args.lr
    OUTPUT_DIR = args.output_dir
    MODEL_LOAD_DIR = args.model_load_dir
    MODEL_SAVE_DIR = args.model_save_dir
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    MEAN = [0.5,0.5,0.5]
    STD = [0.5,0.5,0.5]

    seed_everything(SEED)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    print(MODEL_SAVE_DIR)

    df = pd.read_csv(os.path.join(ROOT_DIR,'train.csv'))
    df["kfold"] = -1  
    df = df.sample(frac=1).reset_index(drop=True)  
    y = df['digit_sum'].values  
    kf = StratifiedKFold(n_splits=5)  
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):  
        df.loc[v_, 'kfold'] = f  

    train_df = df[df["kfold"]!=0]
    val_df = df[df["kfold"]==0]
    test_files = glob(f'{ROOT_DIR}/test/*')

    transform = get_transform(IMAGE_SIZE,
                            MEAN,
                            STD)

    train_dataset = UMNIST(train_df,os.path.join(ROOT_DIR,"train"),transform)
    val_dataset = UMNIST(val_df,os.path.join(ROOT_DIR,"train"),transform)
    test_dataset = UMNIST_test(test_files,transform)


    model = mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(1280,NUM_CLASSES,bias=True)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    steps_per_epoch = len(train_dataset)//(TRAIN_BATCH_SIZE)
    if len(train_dataset) % TRAIN_BATCH_SIZE != 0:
        steps_per_epoch += 1
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, WARMUP_EPOCHS*steps_per_epoch, EPOCHS*steps_per_epoch)

    for param in model.parameters():
        param.requires_grad = True



    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        infer_batch_size=INFER_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        epochs=EPOCHS,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_save_dir=MODEL_SAVE_DIR,
        model_load_dir=MODEL_LOAD_DIR,
        test_output_dir=OUTPUT_DIR
    )

    trainer.run()
    


INFERENCE_TIME_CAP = 120 # 120 sec

args_train = {
    'root_dir': '/kaggle/input/budgeted-model-inference-iccv-2023-rcv-workshop/umnist_iccv_1024/umnist_iccv_1024',
    'epochs': 100,
    'train_batch_size': 6,
    'script_location': '/kaggle/usr/lib/starter_inference_script/starter_inference_script.py',
    'time': 8.8*60*60
       }
args_test = {
    'root_dir': '/kaggle/input/budgeted-model-inference-iccv-2023-rcv-workshop/umnist_iccv_1024/umnist_iccv_1024',
    'epochs': 0,
    'train_batch_size': 6,
    'model_load_dir': './models',
    'script_location': '/kaggle/usr/lib/starter_inference_script/starter_inference_script.py',
    'time': INFERENCE_TIME_CAP
       }



def terminate_subprocess(process):
    print("Terminating subprocess...")
    process.terminate()
    
    
def main_function(args):
    root_dir = args['root_dir']
    epochs = args['epochs']
    script_location = args['script_location']
    train_batch_size = args['train_batch_size']
    model_load_dir = None
    if 'model_load_dir' in args.keys():
        model_load_dir = args['model_load_dir']
    # Call caller.py with optional arguments using subprocess.Popen
    if model_load_dir is None:
        caller_process = subprocess.Popen(
            ["python",script_location, '--root_dir', root_dir,'--train_batch_size', str(train_batch_size),  "--epochs", str(epochs)])
    else:
        caller_process = subprocess.Popen(
            ["python",script_location, '--root_dir', root_dir, '--train_batch_size', str(train_batch_size),'--model_load_dir',model_load_dir, "--epochs", str(epochs)])
    # Start a timer to terminate the subprocess after a certain time 
    exit_timer = threading.Timer(
        args['time'], terminate_subprocess, args=[caller_process])
    exit_timer.start()

    # Wait for the subprocess to complete
    caller_process.wait()

if __name__ == '__main__':
    main_function(args_train)
    main_function(args_test)