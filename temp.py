import data_module
import numpy as np
data_obj = data_module.VertexNumericData(None, ['id', 'attribute_1', 'attribute_2'])
data_obj.num_files = 1
data_obj.num_vertices = 10
id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int)
attribute_1 = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1])
attribute_2 = np.array([2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9])
id = np.reshape(id, (10, 1))
attribute_2 = np.reshape(attribute_2, (10, 1))
attribute_1 = np.reshape(attribute_1, (10, 1))
data_obj.data = np.concatenate([id, attribute_1, attribute_2], axis=1)
data_obj.attribute_data = np.concatenate((attribute_1, attribute_2), axis=1)
data_obj.attribute_keys = ['attribute_1', 'attribute_2']
data_obj.normalize_attribute_data()
data_obj.principal_component_analysis()

print('Holding')

