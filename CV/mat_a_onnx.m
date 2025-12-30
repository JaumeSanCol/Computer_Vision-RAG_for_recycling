% Cargamos el modelo entrenado que hemos guardado en un formato accesible
% para matlab
load('ModeloReciclarIA_Entrenado.mat'); 
% Guardamos en una variable la red
mi_red = net; 
% Definimos el nombre del nuevo formato
outputFile = 'cv_reciclaria.onnx';
% Exportamos a ONNX, un fomrato 'universal'
exportONNXNetwork(mi_red, outputFile);
fprintf('El archivo se guardó como: %s\n', outputFile);