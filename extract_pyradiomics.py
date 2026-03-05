import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# 设置数据路径
image_dir = r'W:\STAS_multis_new\dierpi_bu\dicom'
label_dir = r'W:\STAS_multis_new\dierpi_bu\labels'
output_csv = r'W:\STAS_multis_new\dierpi_bu\diepinostasfeatures1.csv'

# 获取所有病例的文件名
image_files = [f for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
label_files = [f for f in os.listdir(label_dir) if f.endswith('.nii')] #.mask

# 确保每个病例的图像和标签文件匹配
image_files.sort()
label_files.sort()

# 将列表转换为 DataFrame，并命名这一列为 filename
dfs = pd.DataFrame({'filename': image_files})

# 保存到 CSV 文件，不保留索引
files_csv = r"W:\STAS_multis_new\dierpi_bu\diepinostasfeatures_list1.csv"
dfs.to_csv(files_csv, index=False)
# 创建特征提取器
params = 'M:\\STAS_multis\\Params.yaml'  # 如果有自定义参数文件
# settings = {'binWidth':25, 'sigma':[3,5], 'Interpolator': sitk.sitkBSpline,
#                 'resampledPixelSpacing':[1,1,1], 'voxelArrayShift':1000,
#                 'normalize':True, 'normalizeScale':100, 'correctMask': True, 'geometryTolerance': 1e-3}
extractor = featureextractor.RadiomicsFeatureExtractor()#(**settings)#(params)

# 初始化结果列表
results = []
df=pd.DataFrame()# 创建空 DataFrame 用于存储特征
# 遍历每个病例
# i = 1
for image_file, label_file in zip(image_files, label_files):

    # i = i+1
    print(image_file)
    # image_file ='0013006242.image.nii.gz'
    # label_file = '0013006242.nii'
    image_path = os.path.join(image_dir, image_file)
    # 获取文件名（去掉路径），然后去掉 .nii.gz 后缀并添加 .nii 后缀
    file_name = os.path.basename(image_path)  # 获取文件名: 'CT01056647.nii.gz'
    new_file_name = file_name.replace('.nii.gz', '.nii')  # 替换后缀

    # 获取文件夹路径
    folder_path = os.path.dirname(image_path)  # 获取文件夹路径: 'W:\\STAS_multis_new\\dierpi_bu\\dicom'

    # 构造新路径
    label_path = os.path.join(r'W:\STAS_multis_new\dierpi_bu\labels', new_file_name)
    # label_path = os.path.join(label_dir, label_file)

    # 读取图像和标签
    image = sitk.ReadImage(image_path)
    label = sitk.ReadImage(label_path)

    # 提取特征
    feature_vector = extractor.execute(image, label)
    extractor = featureextractor.RadiomicsFeatureExtractor()  # 创建一个特征提取器对象，用于从影像中提取特征
    # extractor.settings['geometryTolerance'] = 1e-5
    # 指定要提取的影像类型，这里启用了多个影像变换类型，可以根据需要进行删减或者添加
    extractor.enableImageTypes(Original={}, Exponential={}, Gradient={}, Logarithm={}, Square={}, SquareRoot={},
                               Wavelet={})
    # extractor.settings['correctMask'] = True
    # extractor.settings['geometryTolerance'] = 1e-5  # 可配合使用
    featureVector = extractor.execute(image, label)

    # featureVector = extractor.execute(image, label)  # 使用影像文件和掩膜文件来提取特征
    # 将提取的特征转换为 DataFrame
    df_new = pd.DataFrame([featureVector])  # 直接将 featureVector 转换为 DataFrame
    df_new.columns = featureVector.keys()  # 将 featureVector 中的特征名称作为 DataFrame 的列名
    # df_new.insert(loc=0, column='image_file', value=image_files)
    df = pd.concat([df, df_new], ignore_index=True)
    # # 将特征添加到结果列表
    # feature_dict = {'Image': image_file, 'Label': label_file}
    # feature_dict.update(feature_vector)
    # results.append(feature_dict)
    print(len(df))

# # 将结果转换为 DataFrame
# df = pd.DataFrame(results)
#     if i ==4:
        # 保存为 CSV 文件
df.to_csv(output_csv, index=False)

print(f"特征已成功保存到 {output_csv}")
