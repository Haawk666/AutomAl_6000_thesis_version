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
btn_export = 'Export .png of the overlay image'
btn_start_alg_1 = 'Start column detection'
btn_reset_alg_1 = 'Restart column detection'
btn_start_alg_2 = 'Start column characterization'
btn_reset_alg_2 = 'Restart column characterization'
btn_invert_lvl_alg_2 = 'Invert all z-heights'
btn_delete = 'Not implemented'
btn_sub = 'Generate 1. order sub-graph seeded on the currently selected column'
btn_deselect = 'Deselct any columns'
btn_new = 'Not implemented'
btn_set_style = ''
btn_set_indices = ''
btn_set_indices_2 = ''
btn_set_perturb_mode = ''
btn_plot_variance = ''
btn_plot_angles = ''

chb_precipitate_column = ''
chb_show = ''
chb_move = ''
chb_graph = 'Toggle edges that are not reciprocated'
chb_raw_image = 'Toggle the raw image in the background'
chb_black_background = ''
chb_structures = ''
chb_boarders = ''
chb_si_columns = ''
chb_si_network = ''
chb_mg_columns = ''
chb_mg_network = ''
chb_al_columns = ''
chb_al_network = ''
chb_cu_columns = ''
chb_cu_network = ''
chb_ag_columns = ''
chb_ag_network = ''
chb_un_columns = ''
chb_columns = ''
chb_al_mesh = ''
chb_neighbours = ''
chb_legend = ''
chb_scalebar = ''

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
                           btn_sub,
                           btn_deselect,
                           btn_new,
                           btn_set_style,
                           btn_set_indices,
                           btn_set_indices_2,
                           btn_set_perturb_mode,
                           btn_plot_variance,
                           btn_plot_angles]

control_window_chb_list = [chb_precipitate_column,
                           chb_show,
                           chb_move,
                           chb_graph,
                           chb_raw_image,
                           chb_black_background,
                           chb_structures,
                           chb_boarders,
                           chb_si_columns,
                           chb_si_network,
                           chb_mg_columns,
                           chb_mg_network,
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
