import numpy as np
import matplotlib.pyplot as plt
import mne, mne_connectivity

def plot_connectivity(con_mat, labels, con_methods, subjects_dir, subject, save_fig=True):    

    # First, we reorder the labels based on their location in the left hemi
    label_names = [label.name for label in labels]
    label_colors = [label.color for label in labels]
    
    lh_labels = [name for name in label_names if name.endswith('lh')]
    
    # Get the y-location of the label
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)
    
    # Reorder the labels based on their location
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
    
    # For the right hemi
    rh_labels = [label[:-2] + 'rh' for label in lh_labels]
    
    # Save the plot order and create a circular layout
    node_order = list()
    node_order.extend(lh_labels[::-1])  # reverse the order
    node_order.extend(rh_labels)
    
    node_angles = mne.viz.circular_layout(label_names, node_order, start_pos=90,
                                          group_boundaries=[0, len(label_names) / 2])

    # Plot the graph using node colors from the FreeSurfer parcellation.
    if save_fig==False:
        show = True
    else:
        show = False
    
    for i in con_methods:
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                               subplot_kw=dict(polar=True))
        mne_connectivity.viz.plot_connectivity_circle(con_mat[i], label_names, n_lines=100,
                                                      node_angles=node_angles, node_colors=label_colors,
                                                      title='All-to-All Connectivity [%s]' % i,
                                                      ax=ax, show=show)
        fig.tight_layout()
        
        if save_fig == True:
            # Get a connectivity results folder in subject dir
            results_folder = os.path.join(subjects_dir, subject, 'connectivity_results')
            if not os.path.exists(results_folder):
                os.system('mkdir' % results_folder)
            fig.savefig()
        