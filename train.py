import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import GPUtil
import datetime
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score, \
    classification_report
from utils.dataloader import setDataLoader
from utils.logger import  Logger
from utils.loss import ETF_Loss
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

def cal_flops_and_parameter(model):
    # 分析FLOPs
    tensor = torch.zeros(( 1, 68, 1024)).cuda()
    flops = FlopCountAnalysis(model, tensor)
    print("FLOPs: ", flops.total()," F,",flops.total()/1e9," G")
    # 分析parameters
    print("Param: ", parameter_count_table(model))

use_gpu = torch.cuda.is_available()
torch.cuda.manual_seed(3407)
if (use_gpu):
    deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False, excludeID=[],
                                    excludeUUID=[])
    if (len(deviceIDs) != 0):
        deviceIDs = GPUtil.getAvailable(order='first', limit=1, maxLoad=1, maxMemory=1, includeNan=False, excludeID=[],
                                        excludeUUID=[])
        print(deviceIDs)
        print("detect set :", deviceIDs)
        device = torch.device("cuda:" + str(deviceIDs[0]))
else:
    device = torch.device("cpu")
print("use gpu:", use_gpu)

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Train():
    def __init__(self, 
            in_channles,  name='dense', method_type=0, is_show=True, 
            is_drop = False, 
            token_type = "ALL"
        ):
        self.in_channels = in_channles

        self.name = name
        self.method_type = method_type
        self.token_type = 0 if token_type == "ALL" else 1
        self.lr = 0.0001
        self.is_drop = is_drop
        self.history_acc = []
        self.history_loss = []
        self.history_test_acc = []
        self.history_test_loss = []
        self.create(is_show)

    def create(self, is_show):
        hidden_dim = 80 # 64 128 256 448 512 
        n_head = 8 # 1 - 8
        n_encoder = 1 # 1 - 4
        n_decoder = 1 # 1 - 4
        
        if (self.method_type == 0):
            from models.MyModel import HRE as Model
            self.model = Model(in_dim=512, hidden_dim=64, n_head=8, n_encoder=1, n_decoder=1, n_query=4, dropout=0.3)
            print("build HRE model")
        elif (self.method_type == 1):
            if self.token_type:
                from models.MyModel import HREC_ as Model
                # 512, 64, 8, 1, 2, 6, 0.3
                # origin
                # self.model = Model(in_dim=512, hidden_dim=64, n_head=8, n_encoder=1, n_decoder=2, n_query=6, dropout=0.3)
                # origin_s 448                 # 64 128 256 512 
                self.model = Model(in_dim=512, hidden_dim=hidden_dim, n_head=n_head, n_encoder=n_encoder, n_decoder=n_decoder, n_query=6, dropout=0.3)
                # small
                # self.model = Model(in_dim=512, hidden_dim=32, n_head=1, n_encoder=1, n_decoder=1, n_query=6, dropout=0.3)
                # large
                # self.model = Model(in_dim=512, hidden_dim=128, n_head=1, n_encoder=1, n_decoder=2, n_query=6, dropout=0.3)
                print("build HREC model token")
            else:
                from models.MyModel import HREC as Model
                # 512, 64, 8, 1, 2, 6, 0.3
                # self.model = Model(in_dim=512, hidden_dim=64, n_head=8, n_encoder=1, n_decoder=2, n_query=6, dropout=0.3)
                # origin_s
                self.model = Model(in_dim=512, hidden_dim=hidden_dim, n_head=n_head, n_encoder=n_encoder, n_decoder=n_decoder, n_query=6, dropout=0.3)
                # small
                # self.model = Model(in_dim=512, hidden_dim=32, n_head=1, n_encoder=1, n_decoder=1, n_query=6, dropout=0.3)
                # large 
                # self.model = Model(in_dim=512, hidden_dim=128, n_head=1, n_encoder=1, n_decoder=2, n_query=6, dropout=0.3)
                print("build HREC model")
        elif (self.method_type == 2):
            if self.token_type:
                from models.dnn import DNN_ as Model
                self.model = Model(in_dim=1)
                print("build dnn model token")
            else:
                from models.dnn import DNN as Model
                self.model = Model(in_dim=1)
                print("build dnn model")
        elif (self.method_type == 3):
            if self.token_type:
                from models.bigrua import BIGRU_Attention_ as Model
                self.model = Model(in_dim=512)
                print("build BIGRU_Attention model token") 
            else:
                from models.bigrua import BIGRU_Attention as Model
                self.model = Model(in_dim=512)
                print("build BIGRU_Attention model") 
        elif (self.method_type == 4):
            if self.token_type:
                from models.bilstmformer import CBiLSTMAF_ as Model
                self.model = Model(in_dim=512 )
                print("build Transformer-CNN bilstm Attention model token") 
            else:
                from models.bilstmformer import CBiLSTMAF as Model
                self.model = Model(in_dim=512 )
                print("build Transformer-CNN bilstm Attention model") 
        else:
            raise NotImplementedError

        self.cost = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss(reduction='mean')
        self.etf = ETF_Loss()
        if (use_gpu):
            self.model = self.model.to(device)
            self.cost = self.cost.to(device)
            self.mse  = self.mse.to(device)
            self.etf = self.etf.to(device)
            
        if (is_show):
            summary(self.model, self.in_channels)
        # cal_flops_and_parameter( self.model )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def train_and_test(self, n_epochs, data_loader_train, data_loader_test):
        best_loss = 1000000
        es = 0
        self.all_y_true = []
        self.all_y_pred = []
        global_best = []
        epoch_train_loss = 0
        for epoch in range(n_epochs):
            start_time = datetime.datetime.now()
            print("Epoch {}/{}".format(epoch, n_epochs))
            print("-" * 10)

            epoch_train_loss = self.train(data_loader_train)
            train_acc = self.compute_overall_metrics()
            epoch_test_loss, best_list = self.test(data_loader_test)
            test_acc = self.compute_overall_metrics()

            self.history_acc.append(0)
            self.history_loss.append(epoch_train_loss)
            self.history_test_acc.append(0)
            self.history_test_loss.append(epoch_test_loss)
            print(
                "Loss is:{:.4f}\nLoss is:{:.4f}\ncost time:{:.4f} min, ETA:{:.4f}".format(
                    epoch_train_loss,
                    epoch_test_loss,
                    (datetime.datetime.now() - start_time).seconds / 60,
                    (n_epochs - 1 - epoch) * (datetime.datetime.now() - start_time).seconds / 60,
                )
            )
            # print("train: ACC is:{:.6f}, PRE is:{:.6f}, Recall is:{:.6f}, F1 is:{:.6f}".format(*train_acc))
            print("test: ACC is:{:.6f}, PRE is:{:.6f}, Recall is:{:.6f}, F1 is:{:.6f}".format(*test_acc))
            F1 = test_acc[0]
            epoch_test_loss = 1 - F1
            if (epoch_test_loss < best_loss):
                best_loss = epoch_test_loss
                global_best = test_acc
                es = 0
                self.save_parameter("./save_best/", "best", best_list)
                
            else:
                es += 1
                print("Counter {} of 20".format(es))
                if es > 10 and epoch > 50:
                    print("Early stopping with F1: ", 1 - best_loss, "and F1 for this epoch: ", 1 -epoch_test_loss,
                          "...")
                  
                    break
        print(global_best)
        self.save_history()
        self.save_parameter()

    def test(self, data_loader_test):
        self.model.eval()
        running_loss = 0
        test_index = 0
        pred_list = []
        gt_list = []
        q_list = []
        with torch.no_grad():
            for data in data_loader_test:
                x = self.unpackage(data)
                label = x[-1]

                outputs = self.model(x[:-1])
                if type(outputs[0]) is list:
                    loss = self.cost(outputs[0][0], label)
                    coef_matrix = outputs[0][1]
                    loss += self.etf( coef_matrix, label)
                else:
                    loss = self.cost(outputs[0], label)

                if type(outputs[0]) is list:
                    self.calculate_metrics(label, outputs[0][0])
                else:
                    self.calculate_metrics(label, outputs[0])
                # self.calculate_metrics(label, outputs[0])

                
                running_loss += loss.data.item()
                test_index += 1
                if type(outputs[0]) is list:
                    pred_list.append(outputs[0][0].detach().cpu().numpy())
                else:
                    pred_list.append(outputs[0].detach().cpu().numpy())
                q_list.append(outputs[1].detach().cpu().numpy())
                gt_list.append(label.detach().cpu().numpy())
                # print( len(pred_list), outputs[0].shape )
                # print( len(q_list), outputs[1].shape )
        epoch_loss = running_loss / (test_index + 1)
        return epoch_loss, [pred_list, gt_list, q_list]

    def train(self, data_loader_train):
        self.model.train()
        train_index = 0
        running_loss = 0.0

        for data in data_loader_train:
            x = self.unpackage(data, self.is_drop)

            label = x[-1]
            self.optimizer.zero_grad()
            
            outputs = self.model(x[:-1])
            if type(outputs[0]) is list:
                loss = self.cost(outputs[0][0], label) + 0.01 * self.mse( outputs[0][0], label )
                coef_matrix = outputs[0][1]
                loss += self.etf( coef_matrix, label)
            else:
                # print("ddnn")
                loss = self.cost(outputs[0], label)
            # loss = self.cost(outputs[0], label) # + self.mse( outputs[0], label )

            loss.backward()
            self.optimizer.step()

            running_loss += loss.data.item()
            train_index += 1

            if type(outputs[0]) is list:
                self.calculate_metrics(label, outputs[0][0])
            else:
                self.calculate_metrics(label, outputs[0])

        epoch_train_loss = running_loss / train_index
        return epoch_train_loss

    def calculate_metrics(self, y_true, y_pred):
   
        gt = np.argmax(y_true.detach().cpu().numpy(), 1)
        pred = np.argmax(y_pred.detach().cpu().numpy(), 1)

        self.all_y_true += gt.tolist()
        self.all_y_pred += pred.tolist()

    def compute_overall_metrics(self):
        accuracy = accuracy_score(self.all_y_true, self.all_y_pred)
        precision = precision_score(self.all_y_true, self.all_y_pred, average='macro')
        recall = recall_score(self.all_y_true, self.all_y_pred , average='macro')
        f1 = f1_score(self.all_y_true, self.all_y_pred, average='macro')

        # 清空累积的列表，为下一轮训练/测试准备
        self.all_y_true = []
        self.all_y_pred = []

        return accuracy, precision, recall, f1
    def unpackage(self, data, use_drop=False):
        if self.token_type :
            x, label = data
            if (use_gpu):
                x = x.to(device)
                label = label.to(device)
            return x, label

        else:

            parameters, source_code, ast, ast_deep, node_type_statistics, ast_total, ast_width, entropy, label = data
            if (use_gpu):
                parameters = parameters.to(device)
                source_code = source_code.to(device)
                ast = ast.to(device)
                ast_deep = ast_deep.to(device)

                node_type_statistics = node_type_statistics.to(device)
                ast_total = ast_total.to(device)
                ast_width = ast_width.to(device)
                entropy = entropy.to(device)

                label = label.to(device)

            if use_drop:
                if np.random.randn(1) < 0.95:
                    code_index = np.random.randint(4)
                    attr_index = np.random.randint(4)
                    if code_index == 0:
                        parameters = torch.ones_like(parameters) * -1
                    if code_index == 1:
                        source_code = torch.ones_like(source_code) * -1
                    if code_index == 2:
                        ast = torch.ones_like(ast) * -1
                    if code_index == 3:
                        ast_deep = torch.ones_like(ast_deep) * -1

                    if attr_index == 0:
                        node_type_statistics = torch.ones_like(node_type_statistics) * -1
                    if attr_index == 1:
                        ast_total = torch.ones_like(ast_total) * -1
                    if attr_index == 2:
                        ast_width = torch.ones_like(ast_width) * -1
                    if attr_index == 3:
                        entropy = torch.ones_like(entropy) * -1

            return parameters, source_code, ast, ast_deep, node_type_statistics, ast_total, \
                ast_width, entropy, label
    def save_history(self, file_path='./save/'):
        file_path = file_path + self.name + "/"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        fo = open(file_path + "loss_history.txt", "w+")
        fo.write(str(self.history_loss))
        fo.close()
        fo = open(file_path + "acc_history.txt", "w+")
        fo.write(str(self.history_acc))
        fo.close()
        fo = open(file_path + "loss_test_history.txt", "w+")
        fo.write(str(self.history_test_loss))
        fo.close()
        fo = open(file_path + "test_history.txt", "w+")
        fo.write(str(self.history_test_acc))
        fo.close()

    def save_parameter(self, file_path='./save/', name=None, best_list=None):
        file_path = file_path + self.name + "/"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        if name == None:
            file_path = file_path + "model_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(
                "-", "_").replace(".", "_") + ".pkl"
        else:
            file_path = file_path + name + ".pkl"
        
        if best_list != None:
            pred = best_list[0]
            gt = best_list[1]
            q1 = best_list[2]
            arrays_dict = {'arr_0': pred, 'arr_1': gt, 'arr_2': q1}
            np.savez(file_path + "best.npz", **arrays_dict)
        torch.save(obj=self.model.state_dict(), f=file_path)

    def load_parameter(self, file_path='./save/'):
        self.model.load_state_dict(torch.load(file_path))

    

if __name__ == "__main__":
    
    logger = Logger(
        file_name=f"./logs/Transformer_CNN_after.txt",
        file_mode="w+",
        should_flush=True
    )
    batch_size = 1
    train_path = r"/T2004100/data/maliciousCode/code.csv"
    # 指定模型路径
    model_path = "./models/BERT"
    word2vec_path = "./utils/"

    train_loader, validate_loader = setDataLoader(train_path)

    trainer = Train(
        1,
        name="Transformer_CNN_after",
        method_type=4,
        is_show=False
    )
    trainer.train_and_test(100, train_loader, test_loader)
