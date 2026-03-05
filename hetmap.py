# import h5py
# import shap
# import torch
# import shap
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from Models import our as mil
# from scipy import interpolate
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# import h5py
import sys, argparse
# colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]])
# def _get_stride(coordinates: np.ndarray) -> int:
#     xs = sorted(set(coordinates[:, 0]))
#     x_strides = np.subtract(xs[1:], xs[:-1])
#
#     ys = sorted(set(coordinates[:, 1]))
#     y_strides = np.subtract(ys[1:], ys[:-1])
#
#     stride = min(*x_strides, *y_strides)
#
#     return stride
#
# def _visualize_activation_map(
#     activations: np.ndarray,
#     colors: np.ndarray,
#     alpha: float = 1.0,
#     clipping: bool = True,
# ) -> np.ndarray:
#     """Transforms an activation map into an RGBA numpy array.
#     Args:
#         activations: An (h, w, D) array of activations.
#         colors: A (D, 3) array mapping each of the target classes to a color.
#     Returns:
#         An interpolated activation map. Regions which the algorithm assumes to be background
#         will be transparent.
#     """
#     assert colors.shape[1] == 3, "expected color map to have three color channels"
#     assert (
#         colors.shape[0] == activations.shape[2]
#     ), "one color map entry per class required"
#     # activations should be less or equal to 1
#     assert (
#         activations[2].max() <= 1
#     ), f"Activations should be less than one, otherwise maps get clipped! \n \
#         Max value provided {activations[2].max()}."
#     # transform activation map into RGB map
#     rgbmap = activations.dot(colors)
#     # TODO this is a cheap fix only!
#     if clipping:
#         max_cvalue = np.amax(rgbmap)
#         if max_cvalue > 255.0:
#             # Rescale
#             rgbmap = rgbmap / max_cvalue * 255.0
#             print(f"Rescaled rgbmap as max pixel value is {max_cvalue}")
#             print(f"This could potentially be a problem!")
#
#     # create RGBA map with non-zero activations being the foreground
#     mask = activations.any(axis=2)
#
#     # mask * alpha gives alpha at non zero values of activation
#     # below gives value for the alpha channel
#     im_data = (
#         np.concatenate([rgbmap, np.expand_dims(mask * alpha, -1)], axis=2) * 255.5
#     ).astype(np.uint8)
#
#     return im_data
#
#
# def _MIL_heatmap_for_slide(
#     coords: np.ndarray, scores: np.ndarray, colours: np.ndarray = None
# ) -> np.ndarray:
#     """
#     Args:
#         h5_feature_path: path to .h5 file with features to analyse
#         model: model to analyse slide with
#         categories: TODO
#         map_type: one from ['attention','probability', contribution]
#
#     Returns:
#         Tuple of covered_area, legend, heatmap
#         covered_area: extent in x dimension and extent in y dimension of whole slide image
#         legend: details of legend for plot
#         heatmap: the actual heatmap, z coordinate is activation, x and y are integers from 0 to n_x
#             and 0 to n_y, respectively. To get pixel dimensions multiply x and y by stride
#     """
#
#     # get stride
#     stride =512#stride = _get_stride(coords)
#     scaled_map_coords = coords // stride
#     if colours is not None:
#         pass
#     else:
#         colours = colors
#
#     # make a mask, 1 where coordinates have attention 0 otherwise
#     # ndarray of zeros of dimension max_x * max_y
#     mask = np.zeros(scaled_map_coords.max(0) + 1)
#     # add in ones where we have values
#     for coord in scaled_map_coords:
#         mask[coord[0], coord[1]] = 1
#
#     grid_x, grid_y = np.mgrid[
#         0 : scaled_map_coords[:, 0].max() + 1, 0 : scaled_map_coords[:, 1].max() + 1
#     ]
#
#     # interpolate heatmap over grid
#     if scores.ndim < 2:
#         scores = np.expand_dims(scores, 1)
#     activations = interpolate.griddata(scaled_map_coords, scores, (grid_x, grid_y))
#     activations = np.nan_to_num(activations) * np.expand_dims(mask, 2)
#
#     heatmap = _visualize_activation_map(
#         activations.transpose(1, 0, 2), colours[: activations.shape[-1]]
#     )
#
#     return heatmap
# parser = argparse.ArgumentParser(description='Train our model')
# parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
# parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
# parser.add_argument('--lr', default=1e-5, type=float, help='Initial learning rate [0.0002]')
# parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
# parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
# parser.add_argument('--gpu', type=str, default='3')
# parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
# parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
# parser.add_argument('--dataset', default='xiangya2_p53_2_1_1', type=str, help='Dataset folder name')
# # parser.add_argument('--datasets', default='xiangya2', type=str, help='Dataset folder name')
# parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
# parser.add_argument('--model', default='our', type=str, help='model our')
# parser.add_argument('--hidden_channels', type=int, default=300)
# parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
# parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
# parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
# parser.add_argument('--average', type=bool, default=True,
#                     help='Average the score of max-pooling and bag aggregating')
# parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
# parser.add_argument('--agg', type=str, help='which agg')
# parser.add_argument('--c_path', nargs='+',
#                     default=None, type=str,
#                     help='directory to confounders')  # './datasets_deconf/STAS/train_bag_cls_agnostic_feats_proto_8_transmil.npy'
# # parser.add_argument('--dir', type=str,help='directory to save logs')
#
# args = parser.parse_args()
# # 加载数据
# # feats_TME = os.path.splitext(feats_csv_path)[0] + '_graph.h5'
# feats_TME = 'P:\\lung_cancer\\TME\\1285271-7-HE_graph.h5'#feats_TME.replace('features/256/FEATURES_DIRECTORY/pt_files', 'TME')
# # feats_TME = joblib.load(feats_TME)
# with h5py.File(feats_TME, 'r') as hf:
#     # 读取 node_features
#     node_features = hf['node_features'][:]
#
#     # 读取 edges
#     edges = hf['edges'][:]
#
#     # # 读取 labels
#     # labels = hf['labels'][:]
#
#
# # feats_csv_path1 = #feats_csv_path.replace('256/FEATURES_DIRECTORY/pt_files', 'multi_graph_1')
# feats_csv_path1 ='P:\\lung_cancer\\features\\multi_graph_1\\1285271-7-HE.h5' #feats_csv_path1.replace('.pt', '.h5')
# # feats = joblib.load(feats_csv_path1)#torch.load(feats_csv_path)#joblib.load(feats_csv_path)
# # feats = torch.Tensor(np.array(list(feats.values()))).float()
# with h5py.File(feats_csv_path1, 'r') as hf:
#     # 读取 x_img_256 和对应的 edge
#     x_img_256 = hf['x_img_256'][:]
#     x_img_256_edge = hf['x_img_256_edge'][:]
#
#     # 读取 x_img_512 和对应的 edge
#     x_img_512 = hf['x_img_512'][:]
#     x_img_512_edge = hf['x_img_512_edge'][:]
#
#     # 读取 x_img_1024 和对应的 edge
#     x_img_1024 = hf['x_img_1024'][:]
#     x_img_1024_edge = hf['x_img_1024_edge'][:]
#
# # inputs = [
# #     x_img_256, x_img_512, x_img_1024,
# #     x_img_256_edge, x_img_512_edge, x_img_1024_edge,
# #     node_features.astype(np.float32), edges#.astype(np.float32)
# # ]
# # 将输入转换为模型所需的张量格式，并放置在设备上
# # inputs = [torch.tensor(data).to(device) for data in inputs]
# # 训练模型
# x_img_256 = torch.tensor(x_img_256).to(device)
# x_img_512 = torch.tensor(x_img_512).to(device)
# x_img_1024 = torch.tensor(x_img_1024).to(device)
# x_img_256_edge = torch.tensor(x_img_256_edge).to(device)
# x_img_512_edge = torch.tensor(x_img_512_edge).to(device)
# x_img_1024_edge = torch.tensor(x_img_1024_edge).to(device)
# node_features = torch.tensor(node_features).to(device)
# edges = torch.tensor(edges).to(device)
# import Models.our as mil
# milnet = mil.fusion_model_graph(in_channels=args.feats_size, hidden_channels=args.hidden_channels,
#                                 out_channels=args.num_classes).to(device)
# model = milnet.to(device)
# # model_swim.head = nn.Identity()
# td = torch.load(r'M:/project_P53/lung_cancer/baseline_p53_2/xiangya2_p53_2_1_1_our_None_fulltune/1/2_0.pth')
# # td = torch.load(r'./results/model_graph/SGFbest_model_0.pth',)
# model.load_state_dict(td, strict=False)
# # 模型设为评估模式
# model.eval()
# x_256_score,x_512_score,x_1024_score = milnet(x_img_256, x_img_512, x_img_1024,x_img_256_edge, x_img_512_edge, x_img_1024_edge,node_features.to(torch.float32) , edges)
#
# feats_TME='M:\\project_P53\\lung_cancer\\features\\512\\RESULTS_DIRECTORY\\patches\\1285271-7-HE.h5'
# with h5py.File(feats_TME, 'r') as hf:
#     # 读取 node_features
#     node_coords = hf['coords'][:]
#
# print(node_coords)
#
# # from visualizations.mil_heatmaps import _MIL_heatmap_for_slide
#
# a = _MIL_heatmap_for_slide(x_512_score.detach().cpu().numpy(), node_coords)
# print(a)
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import torch


from PIL import Image
from matplotlib import cm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]])
# Load color map (using 'jet' colormap for heatmap visualization)
colormap = plt.cm.get_cmap('jet')
parser = argparse.ArgumentParser(description='Train our model')
parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
parser.add_argument('--feats_size', default=768, type=int, help='Dimension of the feature size [512]')
parser.add_argument('--lr', default=1e-5, type=float, help='Initial learning rate [0.0002]')
parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs [40|200]')
parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [5e-3]')
parser.add_argument('--weight_decay_conf', default=1e-4, type=float, help='Weight decay [5e-3]')
parser.add_argument('--dataset', default='LUAD_TMB_2_1_1', type=str, help='Dataset folder name')
# parser.add_argument('--datasets', default='xiangya2', type=str, help='Dataset folder name')
parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
parser.add_argument('--model', default='our', type=str, help='model our')
parser.add_argument('--hidden_channels', type=int, default=300)
parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
parser.add_argument('--average', type=bool, default=True,
                    help='Average the score of max-pooling and bag aggregating')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--agg', type=str, help='which agg')
parser.add_argument('--c_path', nargs='+',
                    default=None, type=str,
                    help='directory to confounders')  # './datasets_deconf/STAS/train_bag_cls_agnostic_feats_proto_8_transmil.npy'
parser.add_argument('--dir', type=str,help='directory to save logs')
args = parser.parse_args()

def to_percentiles(scores):
    from scipy.stats import rankdata
    scores = rankdata(scores, 'average')/len(scores) * 100
    return scores
def top_k(scores, k, invert=False):
    if invert:
        top_k_ids=scores.argsort()[:k]
    else:
        top_k_ids=scores.argsort()[::-1][:k]
    return top_k_ids

def sample_rois(scores, coords, k=5, mode='range_sample', seed=1, score_start=0.45, score_end=0.55, top_left=None, bot_right=None):

    if len(scores.shape) == 2:
        scores = scores.flatten()

    scores = to_percentiles(scores)
    if top_left is not None and bot_right is not None:
        scores, coords = screen_coords(scores, coords, top_left, bot_right)

    if mode == 'range_sample':
        sampled_ids = sample_indices(scores, start=score_start, end=score_end, k=k, convert_to_percentile=False, seed=seed)
    elif mode == 'topk':
        sampled_ids = top_k(scores, k, invert=False)
    elif mode == 'reverse_topk':
        sampled_ids = top_k(scores, k, invert=True)
    else:
        raise NotImplementedError
    coords = coords[sampled_ids]
    scores = scores[sampled_ids]

    asset = {'sampled_coords': coords, 'sampled_scores': scores}
    return asset
# Load color map (using 'jet' colormap for heatmap visualization)
colormap = plt.get_cmap('jet')
def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key]
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def _get_stride(coordinates: np.ndarray) -> int:
    xs = sorted(set(coordinates[:, 0]))
    x_strides = np.subtract(xs[1:], xs[:-1])

    ys = sorted(set(coordinates[:, 1]))
    y_strides = np.subtract(ys[1:], ys[:-1])

    stride = min(*x_strides, *y_strides)

    return stride


def _MIL_heatmap_for_slide(
    coords: np.ndarray, scores: np.ndarray, colormap=None
) -> np.ndarray:
    """
    Args:
        coords: Coordinates for each patch in the WSI
        scores: Scores for each patch
        colormap: Colormap to visualize the score contribution
    Returns:
        Heatmap as an RGB numpy array
    """
    stride = 512  # Can be adjusted to suit the actual image size
    scaled_map_coords = coords // stride

    # Make a mask, 1 where coordinates have attention, 0 otherwise
    mask = np.zeros(scaled_map_coords.max(0) + 1)
    for coord in scaled_map_coords:
        mask[coord[0], coord[1]] = 1

    grid_x, grid_y = np.mgrid[
        0 : scaled_map_coords[:, 0].max() + 1, 0 : scaled_map_coords[:, 1].max() + 1
    ]

    if scores.ndim < 2:
        scores = np.expand_dims(scores, 1)
    activations = interpolate.griddata(scaled_map_coords, scores, (grid_x, grid_y))
    activations = np.nan_to_num(activations) * np.expand_dims(mask, 2)

    # Normalize activations to range [0, 1]
    activations_min = activations.min()
    activations_max = activations.max()
    activations_normalized = (activations - activations_min) / (activations_max - activations_min + 1e-8)

    # Apply colormap to normalized activations
    heatmap = (colormap(activations_normalized[..., 0])[:, :, :3] * 255).astype(np.uint8)

    return heatmap

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as cm

def normalize(v, eps=1e-8):
    v = v.astype(np.float32)
    v = v - v.min()
    denom = (v.max() + eps)
    return v / denom

def make_heatmap_from_attention(
    attn: np.ndarray,                 # shape: (n,)
    coords: np.ndarray,               # shape: (n, 2), columns = [x, y] at level-0
    tile_size: int = 256,             # patch/裁块边长（像素，level-0）
    canvas_size: tuple = None, # (H, W) 若不提供，则根据 coords 自动推断
    coords_are_centers: bool = False, # coords 是否为 patch 中心点
    aggregate: str = "sum",           # 'sum' 或 'max'
    downsample: int = 8,              # 将 level-0 尺寸缩小多少倍在画布上成图（加速）
    gaussian_sigma_px: float = None,  # 高斯平滑半径（像素，最终画布坐标系）
):
    """
    返回 heatmap（H, W, float32，范围[0,1]）
    """
    assert attn.ndim == 1 and coords.ndim == 2 and coords.shape[1] == 2
    n = attn.shape[0]
    assert coords.shape[0] == n

    # 归一化注意力以便可视化（你也可以换成softmax或不归一化）
    a = normalize(attn)

    # 推断画布尺寸（在 level-0 像素坐标系）
    if canvas_size is None:
        # 以右下角为 coords + tile_size（若为中心点则加半边）
        half = tile_size // 2 if coords_are_centers else 0
        x_max = int(np.max(coords[:,0] - half) + tile_size)
        y_max = int(np.max(coords[:,1] - half) + tile_size)
        H0, W0 = y_max, x_max
    else:
        H0, W0 = canvas_size

    # 下采样后的画布
    H = max(1, H0 // downsample)
    W = max(1, W0 // downsample)
    canvas = np.zeros((H, W), dtype=np.float32)

    # 将每个 patch 的注意力写入画布
    ts_ds = max(1, tile_size // downsample)

    # 左上角坐标（若 coords 是中心点，则换算为左上角）
    if coords_are_centers:
        tl = coords - (tile_size // 2)
    else:
        tl = coords.copy()

    # 映射到下采样画布
    x0 = (tl[:,0] // downsample).astype(int)
    y0 = (tl[:,1] // downsample).astype(int)

    for i in range(n):
        x, y = x0[i], y0[i]
        if x >= W or y >= H or x + ts_ds <= 0 or y + ts_ds <= 0:
            continue
        xs, xe = max(0, x), min(W, x + ts_ds)
        ys, ye = max(0, y), min(H, y + ts_ds)

        if xs >= xe or ys >= ye:
            continue

        if aggregate == "max":
            canvas[ys:ye, xs:xe] = np.maximum(canvas[ys:ye, xs:xe], a[i])
        else:  # 'sum'
            canvas[ys:ye, xs:xe] += a[i]

    # 可选：高斯平滑
    if gaussian_sigma_px is not None and gaussian_sigma_px > 0:
        try:
            from scipy.ndimage import gaussian_filter
            canvas = gaussian_filter(canvas, sigma=gaussian_sigma_px)
        except Exception:
            # 无 scipy 时，用 PIL 的模糊作为退化方案（sigma 不可控，聊胜于无）
            im_tmp = Image.fromarray(normalize(canvas) * 255.0).convert("L")
            im_tmp = im_tmp.filter(ImageFilter.GaussianBlur(radius=2))
            canvas = np.array(im_tmp, dtype=np.float32) / 255.0

    # 归一化到 [0,1]
    canvas = normalize(canvas)
    # 扩为伪彩色（可选）
    heat_rgba = cm.get_cmap("jet")(canvas)  # (H, W, 4), RGBA in [0,1]
    heat_rgb = (heat_rgba[..., :3] * 255).astype(np.uint8)
    return canvas, heat_rgb  # 返回灰度热图和伪彩色热图

import os
from pathlib import Path

folder = Path(r'T:\STAS_multis\data\xiangya2_mutil_graph')
files = [p.stem for p in folder.iterdir() if p.is_file()]

print(files)

# 遍历文件列表，去除文件扩展名
for file in files:
    # # 仅处理文件（排除文件夹等）
    # if os.path.isfile(os.path.join(folder_path, file)):
    #     # 去掉文件后缀
    filename_without_extension = file#os.path.splitext(file)[0]
    filename =file
        # print(filename_without_extension)
    # if filename_without_extension =="1357115-HE":
    #     continue

    # filename_without_extension = [s.split(".")[0] + "_LUAD" for s in filename_without_extension]
    filename1 = filename_without_extension.split(".")[0]#"1409000-8-P"  #"1485428-5-HE"#"TCGA-49-6767-01Z-00-DX1.53459c0e-b8ec-4893-9910-87b63c503134"#"1411193-4-HE"#"1434028-2-HE"#"1430851-3-HE" #"1434028-2-HE"
    # filename1 = filename1 + "_LUAD"
    folder1 = Path(r'T:\STAS_multis\data\xiangya2_mutil_graph')
    files1 = [p.stem for p in folder1.iterdir() if p.is_file()]
    if filename1 in files1:
        print(filename1)
    else:
        continue

    feats_csv_path1 = './data/xiangya2_mutil_graph/{filename}.h5'.format(filename=filename1)#new)  # filename)
    # if not os.path.exists(feats_csv_path1):
    #     continue
    # feats_csv_path1 = 'P:\\lung_cancer\\LUAD_feature\\multi_graph_1\\{filename}.h5'.format(filename=filename.split('.')[0])
    file_path = os.path.join(r"T:\STAS_multis\results\heatmap", filename1 + "_256.jpg")

    # 检查文件是否存在
    if os.path.exists(file_path):
        print(f"文件 {file_path} 已存在，跳过当前循环")
        continue  # 如果文件存在，跳过当前循环

    with h5py.File(feats_csv_path1, 'r') as hf:
        x_img_256 = hf['256'][:]
        coords_256 = hf["256_coords"][()]
        x_img_512 = hf['512'][:]
        coords_512 = hf["512_coords"][()]
        node_features = hf['tme_node'][:]
        # 读取 edges
        edges = hf['tme_edges'][:]
        ct = hf["ct"][()]


    x_img_256 = torch.from_numpy(x_img_256).to(device) #torch.tensor(x_img_256).to(device)
    x_img_512 = torch.from_numpy(x_img_512).to(device) #torch.tensor(x_img_512).to(device)
    tme_node = torch.Tensor(node_features).to(device)
    # tme_node = torch.tensor(node_features).to(device)
    tme_edges = torch.tensor(edges).to(device)
    feats_ct = torch.from_numpy(ct).to(device)

    # import Models.our3 as mil
    # milnet = mil.fusion_model_graph(args = args, in_channels=args.feats_size, hidden_channels=args.hidden_channels, out_channels =args.num_classes).to(device)
    from Models.test_new_models1 import MultiModalMoE_XAttn

    milnet = MultiModalMoE_XAttn(d=256, K=3, d_fuse=256, p=0.2, nhead=4).to(device)
    model = milnet.to(device)
    td = torch.load(r'T:\STAS_multis\baseline_our_2\7\8_3.pth')#our2
    model.load_state_dict(td, strict=False)
    model.eval()

    # 获取模型输出
    results, atten = milnet(x_img_256, x_img_512, tme_node, tme_edges, feats_ct, modality_dropout_p=4)
    x_256_scores = atten['att20'].detach().cpu().numpy()
    x_512_scores = atten['att10'].detach().cpu().numpy()
    # x_256_score,x_512_score, TME_fea = model(x_img_256, x_img_512, x_img_256_edge, x_img_512_edge, node_features.to(torch.float32), edges)
    # scores = x_512_score.detach().cpu().numpy()
    # 假设：attn.shape=(n,), coords.shape=(n,2)，单位为 level-0 像素的左上角
    # canvas, heat_rgb = make_heatmap_from_attention(
    #     attn=x_256_scores,
    #     coords=coords_256,
    #     tile_size=512,
    #     downsample=8,
    #     coords_are_centers=False,
    #     aggregate="sum",
    #     gaussian_sigma_px=2.0,
    # )
    # Image.fromarray(heat_rgb).save('./results/heatmap/{filename}_512.jpg'.format(filename=filename), quality=100)


    # 读取 WSI 图像坐标
    # feats_TME = 'M:\\project_P53\\lung_cancer\\features\\256\\RESULTS_DIRECTORY\\patches\\{filename}.h5'.format(filename=filename)
    # with h5py.File(feats_TME, 'r') as hf:
    #     coords = hf['coords'][:]
    ############################10x热力图
    # from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
    #
    # heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': 1, 'blur': False, 'custom_downsample': 1}
    # vis_patch_size = (1024, 1024)
    # slide_path = 'T:\\features\\all_svs\\{filename}.svs'.format(filename=filename)#'S:\\stas2025\\STAS_2025_all\\{filename}.svs'.format(filename=filename) ##1285271-7-HE
    # if not os.path.exists(slide_path):
    #     continue
    # mask_file = './results/heatmap/{filename}_mask.pkl'.format(filename=filename)
    # seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False} #,'keep_ids': ' ', 'exclude_ids':' '
    # filter_params = {'a_t':1, 'a_h':1, 'max_n_holes':2}
    #
    #
    # # seg_params = load_params(process_stack.loc[i], seg_params)
    # # filter_params = load_params(process_stack.loc[i], filter_params)
    # # vis_params = load_params(process_stack.loc[i], vis_params)
    #
    # import os
    # wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
    # sample = {'k': 0, 'mode': 'topk', 'name': 'topk_high_attention', 'sample': True, 'seed': 1}
    # sample_save_dir =  './results/topk_high_attention/512/{filename}'.format(filename=filename)   #os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
    # os.makedirs(sample_save_dir, exist_ok=True)
    # print('sampling {}'.format(sample['name']))
    # sample_results = sample_rois(x_512_scores, coords_512, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
    #     score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
    # for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
    #     print('coord: {} score: {:.3f}'.format(s_coord, s_score))
    #     patch = wsi_object.wsi.read_region(tuple(s_coord), 0, (512, 512)).convert('RGB')
    #     dpi = 600
    #     patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, filename, s_coord[0], s_coord[1], s_score)), dpi=(dpi, dpi))
    # from matplotlib.colors import ListedColormap
    # cmap = 'jet' # ListedColormap(['blue', 'yellow'])#'jet'
    # heatmap = drawHeatmap(x_512_scores, coords_512, slide_path, wsi_object=wsi_object,
    #                                   cmap=cmap, alpha=0.4, **heatmap_vis_args,
    #                                   binarize=False,
    #                                   blank_canvas=False,
    #                                   thresh=-1,  patch_size = vis_patch_size,
    #                                   overlap=0.5,
    #                                   top_left=None, bot_right = None)
    #
    # from PIL import Image
    #
    # MAX_SIZE = 65000  # 宽或高最大值
    # w, h = heatmap.size
    #
    # # 仅在超大时缩放
    # if w > MAX_SIZE or h > MAX_SIZE:
    #     scale = MAX_SIZE / max(w, h)
    #     new_size = (int(w * scale), int(h * scale))
    #     heatmap = heatmap.resize(new_size, resample=Image.ANTIALIAS)
    #     print(f"图像尺寸过大，已缩放至 {new_size}")
    #
    # heatmap.save('./results/heatmap/{filename}_512.jpg'.format(filename=filename), quality=100)



    ###保存20倍率以下的热力图
    from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches

    heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': 1, 'blur': False, 'custom_downsample': 1}
    vis_patch_size = (512, 512)
    slide_path = 'T:\\features\\all_svs\\{filename}.svs'.format(
        filename=filename)  # 'S:\\stas2025\\STAS_2025_all\\{filename}.svs'.format(filename=filename) ##1285271-7-HE
    if not os.path.exists(slide_path):
        continue
    mask_file = './results/heatmap/{filename}_mask.pkl'.format(filename=filename)
    seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2,
                  'use_otsu': False}  # ,'keep_ids': ' ', 'exclude_ids':' '
    filter_params = {'a_t': 1, 'a_h': 1, 'max_n_holes': 2}

    wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
    sample = {'k': 0, 'mode': 'topk', 'name': 'topk_high_attention', 'sample': True, 'seed': 1}
    sample_save_dir = './results/topk_high_attention/256/{filename}'.format(
        filename=filename)  # os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
    os.makedirs(sample_save_dir, exist_ok=True)
    print('sampling {}'.format(sample['name']))
    sample_results = sample_rois(x_256_scores, coords_256, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
                                 score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
    for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
        print('coord: {} score: {:.3f}'.format(s_coord, s_score))
        patch = wsi_object.wsi.read_region(tuple(s_coord), 0, (256, 256)).convert('RGB')
        dpi = 600
        patch.save(os.path.join(sample_save_dir,
                                '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, filename, s_coord[0], s_coord[1], s_score)),
                   dpi=(dpi, dpi))
    from matplotlib.colors import ListedColormap

    cmap = 'jet'  # ListedColormap(['blue', 'yellow'])#'jet'
    heatmap = drawHeatmap(x_256_scores, coords_256, slide_path, wsi_object=wsi_object,
                          cmap=cmap, alpha=0.4, **heatmap_vis_args,
                          binarize=False,
                          blank_canvas=False,
                          thresh=-1, patch_size=vis_patch_size,
                          overlap=0.5,
                          top_left=None, bot_right=None)

    from PIL import Image

    MAX_SIZE = 65000  # 宽或高最大值
    w, h = heatmap.size

    # 仅在超大时缩放
    if w > MAX_SIZE or h > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        heatmap = heatmap.resize(new_size, resample=Image.ANTIALIAS)
        print(f"图像尺寸过大，已缩放至 {new_size}")

    heatmap.save('./results/heatmap/{filename}_256.jpg'.format(filename=filename), quality=100)











# # 生成热力图
# heatmap = _MIL_heatmap_for_slide(node_coords, x_512_score.detach().cpu().numpy())
#
# # 保存热力图
# heatmap_img = Image.fromarray(heatmap)
# heatmap_img.save("heatmap_wsi.png")
#
# # 可选地显示热力图
# plt.imshow(heatmap)
# plt.axis('off')
# plt.show()
