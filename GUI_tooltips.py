# By Haakon Tvedt @ NTNU
"""Module container for tooltip strings"""


btn_set_threshold_layout = 'Set the threshold value (between 0 and 1). If the maximum pixel value of the search\n' \
                            'matrix is below this value, column detection will stop searching. A good starting\n' \
                            'value to try for most images is for instance 0,3.'
btn_set_search_size_layout = 'Set the maximum number of columns that column detection will try to find.'
btn_set_scale_layout = 'Set the scale of the image in (pm/pixel). This should be automatically set from the .dm3\n' \
                       'metadata, but can be overridden here if needed.'
btn_set_alloy_layout = 'Set the alloy being studied. (Which species that are expected to be present in the image and\n' \
                       'thus used in the model)'
btn_set_start_layout = 'Set a default starting column for the column charaterization. Used to study deterministic\n' \
                       'results when testing the algorithms'
btn_set_std_1_layout = 'DEPRECATED'
btn_set_std_2_layout = 'DEPRECATED'
btn_set_std_3_layout = 'DEPRECATED'
btn_set_std_4_layout = 'DEPRECATED'
btn_set_std_5_layout = 'DEPRECATED'
btn_set_std_8_layout = 'DEPRECATED'
btn_set_cert_threshold_layout = 'DEPRECATED'
btn_find_column_layout = 'Select column by index reference.'
btn_set_species_layout = 'Manually determine the atomic species of the selected column.'
btn_set_level_layout = 'Manually set the z-height of the selected column.'

btn_cancel_move = 'Cancel the column position'
btn_set_move = 'Accept the new column position'

btn_show_stats = 'Show a numerical summary of the image.'
btn_show_source = 'Show the full path and filename of the original image this project was created from\n' \
                  '(in case you forgot;)'
btn_export = 'Export data with the export wizard'
btn_start_alg_1 = 'Start column detection'
btn_reset_alg_1 = 'Restart column detection'
btn_start_alg_2 = 'Start column characterization'
btn_reset_alg_2 = 'Restart column characterization'
btn_invert_lvl_alg_2 = 'Invert all z-heights'
btn_delete = 'Not implemented'
btn_print_details = 'Print a summary of vertex details to the terminal window'
btn_snap = 'Snap the view to the selected column'
btn_sub = 'Generate 1. order sub-graph seeded on the currently selected column'
btn_refresh_mesh = 'Update internal mesh information and graphical representations'
btn_deselect = 'Deselect any columns'
btn_new = 'Not implemented'
btn_set_style = 'Not implemented'
btn_set_indices = 'DEPRECATED. Use \'permute mode\''
btn_test = 'Ad-Hoc debugging functionality'
btn_crash = 'Throw an exception for science!... (Save your work first!)'
btn_plot = 'Open a plotting wizard to generate a range of plots'
btn_print_distances = 'Print a convenient list of inter-atomic hard-sphere distances in the terminal window\n' \
                      'for quick reference'
btn_build_anti_graph = 'Build the \'anti-graph\' of the graph. In an anti-graph only vertices in the same spatial\n' \
                       'plane are connected.'
btn_pca = 'Launch a principle component analysis (PCA) from a wizard'

chb_toggle_positions = 'Toggle the overlay of atomic positions'
chb_show_graphic_updates = 'Determines whether the graphical representation of the data will update while the\n' \
                           'algorithm is running. This will slow down the process significantly, but could\n' \
                           ' potentially make the process more entertaining if you have nothing else to do...'
chb_precipitate_column = 'Toggle weahter the selected column is counted to be part of the particle or matrix.'
chb_show = 'Toggle the apparence of the selected column in the overlay composition.'
chb_move = 'Enable move mode. Use the \'atomic positions\'-tab for this.'
chb_perturb_mode = 'Initiate perturbation mode. Selecting three different columns will have the following\n' \
                       'effect: The second and third selected columns will switch positions in the neighbour\n' \
                       'indices in the first selected column.'
chb_enable_ruler = 'When enabled, the distance between successively selected columns will be displayed in the\n' \
                   'terminal window.'
chb_graph = 'Toggle edges that are not reciprocated'
chb_toggle_mesh = 'Toggle mesh details'
chb_show_level_0 = 'Toggle the 0 -plane.'
chb_show_level_1 = 'Toggle the 1/2 -plane.'
chb_raw_image = 'Toggle the raw image in the background'
chb_black_background = 'Toggle black background'
chb_structures = ''
chb_boarders = 'Toggle particle boarders'
chb_si_columns = 'Toggle all Si-columns'
chb_si_network = ''
chb_mg_columns = 'Toggle al Mg-columns'
chb_particle = 'Toggle particle columns'
chb_al_columns = 'Toggle Al-columns'
chb_al_network = ''
chb_cu_columns = 'Toggle Cu-columns'
chb_cu_network = ''
chb_ag_columns = 'Toggle Ag-columns'
chb_ag_network = ''
chb_un_columns = 'Toggle Un-columns'
chb_columns = 'Toggle all columns'
chb_al_mesh = 'Toggle Al-matrix'
chb_neighbours = 'Toggle neighbour edges'
chb_legend = 'Toggle legend'
chb_scalebar = 'Toggle scalebar'

control_window_set_list = [btn_set_threshold_layout,
                           btn_set_search_size_layout,
                           btn_set_scale_layout,
                           btn_set_alloy_layout,
                           btn_set_start_layout,
                           btn_set_std_1_layout,
                           btn_set_std_2_layout,
                           btn_set_std_3_layout,
                           btn_set_std_4_layout,
                           btn_set_std_5_layout,
                           btn_set_std_8_layout,
                           btn_set_cert_threshold_layout,
                           btn_find_column_layout,
                           btn_set_species_layout,
                           btn_set_level_layout]

control_window_move_list = [btn_cancel_move,
                            btn_set_move]

control_window_btn_list = [btn_show_stats,
                           btn_show_source,
                           btn_export,
                           btn_start_alg_1,
                           btn_reset_alg_1,
                           btn_start_alg_2,
                           btn_reset_alg_2,
                           btn_invert_lvl_alg_2,
                           btn_delete,
                           btn_print_details,
                           btn_snap,
                           btn_sub,
                           btn_refresh_mesh,
                           btn_deselect,
                           btn_new,
                           btn_set_style,
                           btn_set_indices,
                           btn_test,
                           btn_crash,
                           btn_plot,
                           btn_print_distances,
                           btn_build_anti_graph,
                           btn_pca]

control_window_chb_list = [chb_toggle_positions,
                           chb_show_graphic_updates,
                           chb_precipitate_column,
                           chb_show,
                           chb_move,
                           chb_perturb_mode,
                           chb_enable_ruler,
                           chb_graph,
                           chb_toggle_mesh,
                           chb_show_level_0,
                           chb_show_level_1,
                           chb_raw_image,
                           chb_black_background,
                           chb_structures,
                           chb_boarders,
                           chb_si_columns,
                           chb_si_network,
                           chb_mg_columns,
                           chb_particle,
                           chb_al_columns,
                           chb_al_network,
                           chb_cu_columns,
                           chb_cu_network,
                           chb_ag_columns,
                           chb_ag_network,
                           chb_un_columns,
                           chb_columns,
                           chb_al_mesh,
                           chb_neighbours,
                           chb_legend,
                           chb_scalebar]

btn_save_log = 'Save the log contents to text-file.'
btn_clear_log = 'Empty the contents of the log.'

