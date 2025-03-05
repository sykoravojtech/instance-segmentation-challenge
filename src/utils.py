import matplotlib.pyplot as plt 
import numpy as np 
import torch
import cv2


def show_image(image,mask,pred_image = None):
    
    if pred_image == None:
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')
        
    elif pred_image != None :
        
        f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10,5))
        
        ax1.set_title('IMAGE')
        ax1.imshow(image.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax2.set_title('GROUND TRUTH')
        ax2.imshow(mask.permute(1,2,0).squeeze(),cmap = 'gray')
        
        ax3.set_title('MODEL OUTPUT')
        ax3.imshow(pred_image.permute(1,2,0).squeeze(),cmap = 'gray')
        

def cv2_imshow(im):
    # Convert the image from BGR to RGB color space and display it      
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(im_rgb)
    plt.axis('off')  # Turn off axis labels
    plt.show()

def get_total_param_num_of_model(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def print_ordered_dict_results(results):
    """
    Function to print the results in a human-readable format with explanations
    and provide some context on whether the results are good.
    
    Parameters:
    results (OrderedDict): Ordered dictionary containing bbox and segm evaluation metrics.
    """
    def evaluate_performance(ap):
        if ap > 50:
            return "This is considered excellent performance."
        elif 30 <= ap <= 50:
            return "This is considered good performance."
        else:
            return "This is considered a lower performance and may need improvement."

    print("Results Summary:")
    print("-" * 30)

    for category, metrics in results.items():
        if category == 'bbox':
            continue
            print("\nBounding Box Results (bbox):")
        elif category == 'segm':
            print("\nSegmentation Results (segm):")

        for metric, value in metrics.items():
            if metric == 'AP':
                print(f"  - Average Precision (AP): {value:.2f}")
                print("    * AP is the average precision across all IoU thresholds from 0.50 to 0.95.")
                print(f"    {evaluate_performance(value)}")
            elif metric == 'AP50':
                print(f"  - Average Precision at IoU=0.50 (AP50): {value:.2f}")
                print("    * AP50 is the precision at an IoU threshold of 0.50.")
                print(f"    {evaluate_performance(value)}")
            elif metric == 'AP75':
                print(f"  - Average Precision at IoU=0.75 (AP75): {value:.2f}")
                print("    * AP75 is the precision at an IoU threshold of 0.75.")
                print(f"    {evaluate_performance(value)}")
            elif metric == 'APs':
                print(f"  - Average Precision for small objects (APs): {value:.2f}")
                print("    * APs evaluates the performance on small-sized objects.")
                print(f"    {evaluate_performance(value)}")
            elif metric == 'APm':
                print(f"  - Average Precision for medium objects (APm): {value:.2f}")
                print("    * APm evaluates the performance on medium-sized objects.")
                print(f"    {evaluate_performance(value)}")
            elif metric == 'APl':
                print(f"  - Average Precision for large objects (APl): {value:.2f}")
                print("    * APl evaluates the performance on large-sized objects.")
                print(f"    {evaluate_performance(value)}")