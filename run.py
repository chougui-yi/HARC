import argparse
from train import Train, setup_seed
from utils.dataloader import setDataLoader
from utils.logger import  Logger
import torch

parser = argparse.ArgumentParser()

# parser.add_argument('--model-name', type=str, default='action_net', help='name used to save model and logs')
# parser.add_argument('--submodel-name', type=str, default='model', help='ab')
# parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
# parser.add_argument("--test", action='store_true', help="only evaluate, don't train")

parser.add_argument('--epoch', type=int, default=101)
# parser.add_argument('--batch', type=int, default=32)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--momentum', type=float, default=0.9)
# parser.add_argument('--weight-decay', type=float, default=1e-4)


# parser.add_argument('--in_dim', type=int, default=1024)
# parser.add_argument('--hidden_dim', type=int, default=256)
# parser.add_argument('--n_head', type=int, default=1)
# parser.add_argument('--n_encoder', type=int, default=1)
# parser.add_argument('--n_decoder', type=int, default=1)
# parser.add_argument('--n_query', type=int, default=1)

# parser.add_argument('--dropout', type=float, default=0.0)


# new 
parser.add_argument('--log_name', type=str, default="DNN")
parser.add_argument('--method', type=int, default=0)
parser.add_argument('--is_drop', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--token_type', type=str, default="ALL")
args = parser.parse_args()




if __name__ == '__main__':
    
    
    setup_seed(0)
    logger = Logger(
        file_name=f"./logs/{args.log_name}.txt",
        file_mode="w+",
        should_flush=True
    )
    print(args)
    train_path = r"/T2004100/data/hy/maliciousCode/train2.csv"
    test_path = r"/T2004100/data/hy/maliciousCode/test2.csv"
    print( train_path, test_path )
    train_loader, validate_loader = setDataLoader( 
        train_path, 
        test_path, 
        batch_size = args.batch_size, 
        token = args.token_type
    )
    trainer = Train(
        1,
        name = args.log_name,
        method_type= args.method,
        is_show=False,
        is_drop = args.is_drop,
        token_type = args.token_type,
    )

    

    trainer.train_and_test(args.epoch, train_loader, validate_loader)
    
