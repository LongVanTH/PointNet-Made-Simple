import numpy as np
import argparse
import torch
from models import cls_model
from utils import create_dir, viz_pc

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output/cls')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    model = cls_model().to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("successfully loaded checkpoint from {}".format(model_path))

    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label)).to(args.device)

    # Predictions
    with torch.no_grad():
        batch_size = 32
        pred_label = torch.tensor([]).to(args.device)
        for i in range(0, test_data.size()[0], batch_size):
            test_data_batch = test_data[i:i+batch_size].to(args.device)
            pred_label_batch = model(test_data_batch)
            pred_label = torch.cat((pred_label, torch.argmax(pred_label_batch, dim=1)), dim=0)

    # Compute Accuracy
    test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.size()[0])
    print("Test accuracy: {}".format(test_accuracy))

    # Confusion Matrix
    num_classes = 3
    confusion_matrix = torch.zeros(num_classes, num_classes)
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion_matrix[true_class, pred_class] = (pred_label[test_label == true_class] == pred_class).sum().item()
    print("Confusion matrix: \n{}".format(confusion_matrix))

    # Visualize a few random test point clouds (successful and failure cases)
    num_pos_per_class = 3
    num_neg_per_class = 2
    for true_class in range(num_classes):
        idx_pos = torch.where((pred_label == test_label) & (test_label == true_class))[0]
        idx_neg = torch.where((pred_label != test_label) & (test_label == true_class))[0]
        idx_pos = idx_pos[torch.randperm(len(idx_pos))]
        idx_neg = idx_neg[torch.randperm(len(idx_neg))]
        idx_pos = idx_pos[:num_pos_per_class]
        idx_neg = idx_neg[:num_neg_per_class]
        for j in idx_pos:
            viz_pc(test_data[j], f'{args.output_dir}/pos_{true_class}_{j}.gif', args.device)
        for j in idx_neg:
            viz_pc(test_data[j], f'{args.output_dir}/neg_{true_class}_pred_{int(pred_label[j])}_{j}.gif', args.device)
