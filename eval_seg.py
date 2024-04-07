import numpy as np
import argparse
import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")
    parser.add_argument('--viz_best_worst', type=int, default=0, help="number of best and worst cases to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/seg')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    model = seg_model(args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind]).to(args.device)

    # Predictions
    with torch.no_grad():
        batch_size = 32
        pred_label = torch.tensor([]).to(args.device)
        for i in range(0, test_data.size()[0], batch_size):
            test_data_batch = test_data[i:i+batch_size].to(args.device)
            pred_label_batch = model(test_data_batch)
            pred_label = torch.cat((pred_label, torch.argmax(pred_label_batch, dim=2)), dim=0) 

    # Compute Accuracy
    accuracies = pred_label.eq(test_label.data).cpu().sum(dim=1).numpy() / (test_label.size()[1])
    test_accuracy = np.mean(accuracies)
    print("test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], test_label[args.i], "{}/gt_{}.gif".format(args.output_dir, args.i), args.device)
    viz_seg(test_data[args.i], pred_label[args.i], "{}/pred_{}_{}.gif".format(args.output_dir, args.i, accuracies[args.i]), args.device)

    # Visualize best and worst cases
    if args.viz_best_worst != 0:
        sort_idx = np.argsort(accuracies)
        best_cases = sort_idx[::-1][:args.viz_best_worst]
        worst_cases = sort_idx[:args.viz_best_worst]
        for i in range(args.viz_best_worst):
            viz_seg(test_data[best_cases[i]], test_label[best_cases[i]], "{}/best_gt_{}.gif".format(args.output_dir, i), args.device)
            viz_seg(test_data[best_cases[i]], pred_label[best_cases[i]], "{}/best_pred_{}_{}.gif".format(args.output_dir, i, accuracies[best_cases[i]]), args.device)
            viz_seg(test_data[worst_cases[i]], test_label[worst_cases[i]], "{}/worst_gt_{}.gif".format(args.output_dir, i), args.device)
            viz_seg(test_data[worst_cases[i]], pred_label[worst_cases[i]], "{}/worst_pred_{}_{}.gif".format(args.output_dir, i, accuracies[worst_cases[i]]), args.device)
        