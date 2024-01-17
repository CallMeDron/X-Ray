from xray import xray, get_stl_size

get_stl_size('D:/X-Ray/models/helmet.stl')

xray(model_path='D:/X-Ray/models/helmet.stl', result_path='D:/X-Ray/results/helmet.png',
     scale=1.0)

get_stl_size('D:/X-Ray/models/horse.stl')

xray(model_path='D:/X-Ray/models/horse.stl', result_path='D:/X-Ray/results/horse.png',
     scale=7.0, phi_x=1.0, phi_y=1.5, phi_z=2.0)

get_stl_size('D:/X-Ray/models/sphinx.stl')

xray(model_path='D:/X-Ray/models/sphinx.stl', result_path='D:/X-Ray/results/sphinx.png',
     scale=1.7, phi_x=4.0, phi_y=4.0, phi_z=2.0)
