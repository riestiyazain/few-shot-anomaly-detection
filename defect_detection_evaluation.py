from __future__ import print_function
from FewShot_models.manipulate import *
from FewShot_models.training_parallel import *
from FewShot_models.imresize import imresize
import FewShot_models.functions as functions
import FewShot_models.models as models
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import torchvision
import os, sys
import tarfile
from tqdm import tqdm
import urllib.request
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from evaluate_model import precision_recall_f1, compute_confusion_matrix


def defect_detection(input_name_model,test_size, opt):
    dataset = opt.dataset
    scale = int(opt.size_image)
    pos_class = opt.pos_class
    alpha = int(opt.alpha)
    path =  dataset + "_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(opt.num_images)
    if os.path.exists(path)==True:
        xTest_input = np.load(path  +"/"+dataset+"_data_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy")
        yTest_input = np.load(path +"/"+dataset+"_labels_test_" + str(pos_class) + str(scale) +  "_" + str(opt.index_download) + ".npy")
    else:
        if os.path.exists(path) == False:
            print("path not exists")
            exit()
    xTest_input = xTest_input[:test_size]
    yTest_input = yTest_input[:test_size]

    num_samples = xTest_input.shape[0]
    batch_size = 1
    batch_n = num_samples // batch_size
    path = "TrainedModels/" + str(opt.input_name)[:-4] + \
           "/scale_factor=0.750000,alpha=" + str(alpha)
    probs_predictions = []
    real = torch.from_numpy(xTest_input[0]).to(opt.device).unsqueeze(0)
    functions.adjust_scales2image(real, opt)
    scores_per_scale_dict = torch.from_numpy(np.zeros((opt.stop_scale+1,batch_n))).to(opt.device)

    def compute_normalized_dict(scores_per_scale_dict):
        for scale in range(0, opt.stop_scale + 1):
            maxi = torch.max(scores_per_scale_dict[scale])
            mini = torch.min(scores_per_scale_dict[scale])
            scores_per_scale_dict[scale] = (scores_per_scale_dict[scale] - mini) / (maxi - mini)
        return scores_per_scale_dict

    transformations_list = np.load("TrainedModels/" + str(opt.input_name)[:-4] +  "/transformations.npy")

    with torch.no_grad():
        for i in tqdm(range(batch_n)):
            reals = {}
            real = torch.from_numpy(xTest_input[i]).unsqueeze(0).to(opt.device)
            real_untouched = torch.from_numpy(xTest_input[i]).unsqueeze(0).to(opt.device)
            real = functions.norm(real)
            real = real[:, 0:3, :, :]
            functions.adjust_scales2image(real, opt)
            real = imresize(real, opt.scale1, opt)
            for index_image in range(int(opt.num_images)):
                reals[index_image] = []
                reals = functions.creat_reals_pyramid(real, reals, opt,index_image)

            err_total,err_total_avg, err_total_abs = [],[],[]
            for scale_num in range(0, opt.stop_scale+1  , 1):
                opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), opt.size_image)
                opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), opt.size_image)
                netD = models.WDiscriminatorMulti(opt)
                if torch.cuda.device_count() > 1:
                    netD = DataParallelModel(netD, device_ids=opt.device_ids)
                netD.to(opt.device)
                netD.load_state_dict(torch.load('%s/%d/netD.pth' % (path, scale_num), map_location=opt.device))
                netD.eval()

                err_scale = []
                for index_image in range(1):
                    score_image_in_scale = 0
                    reals_transform = []
                    for index_transform, pair in enumerate(transformations_list):
                        real = reals[index_image][scale_num].to(opt.device)
                        if (opt.dataset == "biscuit" or opt.dataset == "mvtec") and opt.add_jiggle_transformation:
                            flag_color, is_flip, tx, ty, k_rotate, jiggle = pair
                            real_augment = apply_augmentation(real, is_flip, tx, ty, k_rotate, flag_color, jiggle).to(opt.device)
                        else:
                            flag_color, is_flip, tx, ty, k_rotate = pair
                            real_augment = apply_augmentation(real, is_flip, tx, ty, k_rotate, flag_color).to(opt.device)
                        real_augment = torch.squeeze(real_augment)
                        reals_transform.append(real_augment)
                    real_transform = torch.stack(reals_transform)
                    output = netD(real_transform)

                    output_shape = output.size()
                    image_width = output_shape[-1]

                    if isinstance(output, list):
                        output = [tens.to(opt.device) for tens in output]
                        output = torch.cat(output).detach()
                    else:
                        output = output.to(opt.device)
                    
                    reshaped_output = output.permute(0, 2, 3, 1).contiguous()
                    shape = reshaped_output.shape
                    reshaped_output = reshaped_output.view(-1, shape[3])
                    reshaped_output = reshaped_output[:, :opt.num_transforms]

                    m = nn.Softmax(dim=1)
                    score_softmax = m(reshaped_output)

                    # visualization happens inside this if block
                    if scale_num == opt.stop_scale:
                        print(f"visualizing... image {i} of {int(batch_n)}")
                        visualization = torch.zeros(opt.num_transforms, image_width, image_width)

                        # reshape scores back to be 54, C, H, W
                        output_softmax = score_softmax.view(opt.num_transforms, image_width, image_width, opt.num_transforms)
                        output_softmax = output_softmax.permute(0, 3, 1, 2).contiguous()

                        # loop over each transform
                        for ith in range(opt.num_transforms):
                            # get the score for the transform so indexing the channel which corresponds to transform patch score and the
                            # first dimension which corresponds to the transformation image given to discriminator
                            ith_map = output_softmax[ith, ith, :, :]
                            # reshape the 2D score map to vector
                            ith_map_reshaped = ith_map.reshape(image_width*image_width)
                            # get fraction of patches
                            num_patches = int(ith_map_reshaped.shape[0] * opt.fraction_defect)
                            # getting least confident patche scores and their indices
                            smallest_values, smallest_indices = torch.topk(ith_map_reshaped, k=num_patches, largest=True) # default False

                            # create zero mask
                            # everything else must be zero except the selected patches
                            ith_map_mask = torch.zeros_like(ith_map_reshaped)

                            # put in the selected values at their index locations
                            ith_map_mask[smallest_indices] = smallest_values

                            # reshape vector back to 2D score map and append channel wise to result tensor
                            # each channel corresponds to the score of each transform
                            visualization[ith, :, :] = ith_map_mask.reshape(image_width, image_width)

                        # sum all the score channel wise to get one aggregated score map
                        visualization_flattened = visualization.sum(dim=0)

                        # normalize the value
                        visualization_flattened = (visualization_flattened - visualization_flattened.min()) / (visualization_flattened.max() - visualization_flattened.min())
                        # convert to 8 bit image
                        visualization_flattened = np.uint8(visualization_flattened.cpu().numpy() * 255)
                        # Get the color map by name:
                        cm = plt.get_cmap('turbo')
                        # Apply the colormap like a function to any array:
                        visualization_flattened = cm(visualization_flattened)
                        # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
                        # # But we want to convert to RGB in uint8 and save it:
                        visualization_flattened = (visualization_flattened[:, :, :3] * 255).astype(np.uint8)
                        im = Image.fromarray(visualization_flattened)

                        # save real ground truth image as well
                        real_im = np.uint8(real_untouched.squeeze().permute(1, 2, 0).cpu().numpy() * 255)
                        real_im = Image.fromarray(real_im)
                        real_im = real_im.resize(im.size)

                        if not os.path.exists(f"{dataset}/{input_name_model}"):
                            os.makedirs(f"{dataset}/{input_name_model}")
                            os.makedirs(f"{dataset}/{input_name_model}/defect_prediction")
                            os.makedirs(f"{dataset}/{input_name_model}/real_image")
                            os.makedirs(f"{dataset}/{input_name_model}/both")

                        defect_vis_path = f"{dataset}/{input_name_model}/defect_prediction/{opt.pos_class}_{opt.num_images}_defect_prediction_{i}.png"
                        im.save(defect_vis_path)

                        real_vis_path = f"{dataset}/{input_name_model}/real_image/{opt.pos_class}_{opt.num_images}_real_image_{i}.png"
                        real_im.save(real_vis_path)

                        both_im = Image.new('RGB', (real_im.width + im.width, real_im.height))
                        both_im.paste(real_im, (0, 0))
                        both_im.paste(im, (real_im.width, 0))

                        both_vis_path = f"{dataset}/{input_name_model}/both/{opt.pos_class}_{opt.num_images}_both_image_{i}.png"
                        both_im.save(both_vis_path)

                    score_all = score_softmax.reshape(opt.num_transforms, -1, opt.num_transforms)


                    for j in range(opt.num_transforms):

                        # the transformed image out of M transformations
                        current_transform = score_all[j]

                        # all softmax probability score corresponding to that transformation
                        # here the HxH has been flattned to a vector 
                        # each column corresponds to M transform predicted probability
                        # each row is a patch
                        # so to get score of a transform for each patch we do [:, j]
                        score_transform = current_transform[:, j]

                        # also we dont count the predicted probability score of other transforms
                        # so when considering transformation number 25
                        # we will only torch.mean() the scores for the 25th index of the softmax vector for all patches

                        sorted_score_transform, indices = torch.sort(score_transform, descending=False, dim=0)
                        num_patches = int(sorted_score_transform.shape[0] * opt.fraction_defect)

                        score_transform = torch.mean(sorted_score_transform[:num_patches])
                        score_image_in_scale += score_transform
                    err_scale.append(score_image_in_scale)
                err_scale = torch.stack(err_scale)
                err = torch.max(err_scale, dim=0)[0]
                err = torch.mean(err).item()
                scores_per_scale_dict[scale_num][i] = (err)
                err_total.append(err)
                del netD
            avg_err_total = np.mean(err_total)
            probs_predictions.append(avg_err_total)

        export_dir = "testing_summary/"
        export_path = export_dir + opt.input_name.split('.')[0]
        
        if (os.path.exists(export_dir)==False):
            os.mkdir(export_dir)

        probs_predictions = np.array(probs_predictions)
        probs_predictions = (probs_predictions - probs_predictions.min()) / (probs_predictions.max() - probs_predictions.min())
        probs_predictions = list(probs_predictions)
        
        with open(export_path+ "_fraction_" + str(opt.fraction_defect) + ".txt", "w") as text_file:
            print(pos_class, "results: ", file=text_file)
            print(" ", file=text_file)
            print("results without norm, without top_k: ", file=text_file)
            auc1 = roc_auc_score(yTest_input, probs_predictions)
            print("roc_auc_score (not normal) all ={}".format(auc1), file=text_file)
            
            precision,recall,f1 = precision_recall_f1(yTest_input,probs_predictions)
            print("average (precision score) = {}".format(np.mean(precision)), file=text_file)
            print("recall score = {}".format(np.mean(recall)), file=text_file)
            print("f1 score = {}".format(f1), file=text_file)

            scores_per_scale_dict_norm = compute_normalized_dict(scores_per_scale_dict)
            scores_per_scale_dict_norm = scores_per_scale_dict_norm.cpu().clone().numpy()
            
            print(" ", file=text_file)
            print("results with normalization ", file=text_file)

            probs_predictions_norm_all = np.mean(scores_per_scale_dict_norm, axis=0)
            auc1 = roc_auc_score(yTest_input, probs_predictions_norm_all)
            print("roc_auc_score T1 normalize all ={}".format(auc1), file=text_file)

            precision_norm, recall_norm, f1_norm = precision_recall_f1(yTest_input,probs_predictions_norm_all)
            print("average (precision score)  normalize all = {}".format(np.mean(precision_norm)), file=text_file)
            print("recall score normalize all  = {}".format(np.mean(recall_norm)), file=text_file)
            print("f1 score normalize all = {}".format(f1_norm), file=text_file)

            conf_matrix = compute_confusion_matrix(yTest_input,probs_predictions_norm_all, opt.threshold, export_path)
            print("confusion matrix = {}".format(conf_matrix), file=text_file)

        with open(export_path + '_score.npy', 'wb') as f:
            np.save(f,probs_predictions)

        with open(export_path + '_normalized_score.npy', 'wb') as f:
            np.save(f,probs_predictions_norm_all)
            
        with open(export_path + '_test_input.npy', 'wb') as f:
            np.save(f,yTest_input)
    path = export_dir + dataset +"_test_scale" + str(scale) + "_" + str(pos_class) + "_" + str(opt.num_images)
    # os.remove(path + "/mvtec_data_test_" + str(pos_class) + str(scale) + "_" + str(opt.index_download) + ".npy")
    # os.remove(path + "/mvtec_labels_test_" + str(pos_class) + str(scale) + "_" + str(opt.index_download) + ".npy")
    del xTest_input, yTest_input
