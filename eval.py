from os import listdir
import numpy as np 
import argparse
from inception import find_relevance, find_relevance_array
from saliency import get_saliency_per_grid
from metrics import err_trajectory,gumbel_max_sample_array,rbp_trajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_dir', type=str, default='generated_images')
    parser.add_argument('-target_image', type=str, default='pandatarget.png')
    parser.add_argument('-metric', type=str, default='rbp')
    parser.add_argument('-trajectory', type=str, default='saliency')
    parser.add_argument('-gamma', type=float, default=0.8)
    parser.add_argument('-n_samples', type=float, default=50)
    args = parser.parse_args()

    saliency_pred = np.array(get_saliency_per_grid(args.image_dir))
    print('saliency',saliency_pred)
    images = [ args.image_dir + "/"+i for i in sorted(listdir(args.image_dir)) ]
    relevance=[]
    print('relevance',str(find_relevance_array(args.target_image,images)))

    total_eval=[]
    for n in range(args.n_samples):
        path=(gumbel_max_sample_array(saliency_pred))
        if args.metric=='rbp':
            if args.trajectory=='order':
                total_eval.append(rbp_trajectory(relevance,list(range(len(listdir(args.image_dir)))),args.gamma))
            elif args.trajectory =='saliency':
                total_eval.append(rbp_trajectory(relevance,path,args.gamma))
        elif args.metric =='err':
            if args.trajectory=='order':
                total_eval.append(err_trajectory(relevance,list(range(len(listdir(args.image_dir)))),args.gamma))
            elif args.trajectory =='saliency':
                total_eval.append(err_trajectory(relevance,path,args.gamma))

    print('The quality of the gird of generated images in '+args.image_dir+' is evaluated as :' + str(np.mean(total_eval)))