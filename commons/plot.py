from matplotlib import pylab as plt
import nibabel as nib
import numpy as np
from config.config_resnet18 import train_image_list

def print2D(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def printXandY(out, label):
    plt.subplot(1, 2, 1)
    plt.imshow(out, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='gray')
    plt.show()

def save_nii(img_arr, name, index=1):
    img = nib.load(train_image_list[index])
    # img = nib.load('./dataset/crossmoda2022_etz_{index}_ceT1.nii.gz'.format(index=index))
    # img = nib.load('./dataset/validation/crossmoda2021_ldn_{index}_hrT2.nii.gz'.format(index=index))
    img_affine = img.affine
    new_img = nib.Nifti1Image(img_arr, img_affine)
    nib.save(new_img, "{name}.nii.gz".format(name=name))
    # nib.Nifti1Image(img_arr, np.eye(4)).to_filename(f'{name}.nii.gz'.format(name=name))
    # img_arr要为int16的nparray


def save_nii_(img_arr, name, path):
    img = nib.load(path)
    # img = nib.load('./dataset/crossmoda2022_etz_{index}_ceT1.nii.gz'.format(index=index))
    # img = nib.load('./dataset/validation/crossmoda2021_ldn_{index}_hrT2.nii.gz'.format(index=index))
    img_affine = img.affine
    new_img = nib.Nifti1Image(img_arr, img_affine)
    nib.save(new_img, "{name}.nii.gz".format(name=name))
    # nib.Nifti1Image(img_arr, np.eye(4)).to_filename(f'{name}.nii.gz'.format(name=name))
    # img_arr要为int16的nparray

def draw(loss, loss2, name):
    x = [range(0, len(loss))]
    x = [[row[i] for row in x] for i in range(len(x[0]))]
    fig, ax = plt.subplots()
    ax.plot(x, loss, color="red", label="loss_train")
    ax.plot(x, loss2, color="blue", label="loss_val")
    ax.set_title(name)
    ax.legend()
    plt.show()

def draw1(loss,  name):
    x = [range(0, len(loss))]
    x = [[row[i] for row in x] for i in range(len(x[0]))]
    fig, ax = plt.subplots()
    ax.plot(x, loss, color="red", label="loss_train")
    ax.set_title(name)
    ax.legend()
    plt.show()

if __name__ == '__main__':
    a =[0.3372550135397393, 0.2787780049054519, 0.24466865720308345, 0.22584379472486352, 0.23291398252805937, 0.20497060267497663, 0.20412806040890838, 0.19992790565542554, 0.19327326345702875, 0.1933094066284273, 0.18944976802753366, 0.17995033299793367, 0.18138785141965616, 0.1727074279004465, 0.16843906158338423, 0.17027316930825295, 0.1625352311960381, 0.16916565216429855, 0.1591498783749083, 0.16132010410175376, 0.16848819148119376, 0.1543203211672928, 0.14889739749386258, 0.15936745788254167, 0.14530556420188234, 0.15199456388211768, 0.14364847253360177, 0.17466524972215944, 0.14250813578457935, 0.14412242953625062, 0.15282480799309586, 0.14117673436260741, 0.1478989747069452, 0.13798068520491538, 0.14739422516330428, 0.14051092539549523, 0.14682925958186388, 0.1314681216305041, 0.13558620939274196, 0.13846694382474475, 0.1326697546907741, 0.12493217234378276, 0.13960319516532446, 0.15987849749786698, 0.1324727368014662, 0.14021286461502314, 0.13142011897719424, 0.12800573778298238, 0.1307637088365205, 0.1483074428068231, 0.12135281865788705, 0.13120921288171541, 0.11568030783825595, 0.13896710579485996, 0.12445367006180079, 0.13733060435270486, 0.1326873207707768, 0.12660865647637326, 0.14127501563938416, 0.19782406136231578, 0.13297618269596412, 0.13831048106531735, 0.12482452420922725, 0.1288744608324993, 0.13920045081202104, 0.12468908423476893, 0.12745138301271136, 0.12822543157507543, 0.16530177057923182, 0.13450092271618222, 0.13707199551003135, 0.14307227342025094, 0.14078198583877605, 0.12969421878781007, 0.13427422938706435, 0.1283230888657272, 0.13475773283077971, 0.1303751461369836, 0.131361230559971, 0.12681044609812292, 0.15904903496899034, 0.12431618968105834, 0.1277881250311823, 0.15246081825993632, 0.13054765752800132, 0.13231899848450784, 0.12779730270900155, 0.13446537777781487, 0.13977334814389114, 0.1668054465730877, 0.12530986997096435, 0.20061440680823897, 0.14778715696023859, 0.1312628102934231, 0.14475729676854351, 0.13156643943132265, 0.13168676294710324, 0.13469739744196768, 0.14826228267148786, 0.12547991991690968, 0.15012320228244946, 0.14494432658766923, 0.14481641342053594, 0.1418073587929425, 0.14873170957941076, 0.15138057570742525, 0.14215550796412255, 0.15509809756084628, 0.14993764476283736, 0.14874672419998958, 0.1518372178239667, 0.16045586197920467, 0.1563356511456811, 0.16434551836193903, 0.15849729518041664, 0.1605671636517281, 0.15735287794276423, 0.16448259896234327, 0.1705496694649691, 0.18244937139198833, 0.16061936982947847, 0.15771228277488894, 0.15176195894246516, 0.16440552533806665, 0.15138181732238634, 0.1577655294019243, 0.16167407993065275, 0.14222850990684135, 0.15227003241686718, 0.15015594705777324, 0.14922944504929625, 0.1599880962268166, 0.1620900337786778, 0.1553067333753342, 0.15199399099725744, 0.14904129063791555, 0.15189209491338418, 0.17056955874938032, 0.1449855378140574, 0.1460244035874696, 0.14779635255589432, 0.15418617444052157, 0.15393599414307138, 0.15274990669897068, 0.15019597356086192, 0.15215025388676187, 0.16255796878882076, 0.14666526691745158, 0.16526521037778127, 0.15615874366915744, 0.16600836762834503, 0.15278031915912163, 0.14456448566330515, 0.15168421713473357, 0.14276462550396504, 0.15937383031553548, 0.1469502530425139, 0.15599503335745438, 0.1531779461985697, 0.14200312393191067, 0.15432366801668768, 0.13957693018829045, 0.1569392773039315, 0.1397895744556318, 0.14917506795862448, 0.14928651516042327, 0.15340407362774663, 0.14163371313201345, 0.14093836376686458, 0.14941835962235928, 0.14928674778860548, 0.1735341699955904, 0.1397560404451645, 0.14689343722294207, 0.14693417242201773, 0.15154739716292723, 0.14799890995187603, 0.15122035317851798, 0.16051790806586327, 0.14503602416295072, 0.14505064544146476, 0.14239652547985315, 0.1399009612834324, 0.15669144827710546, 0.1510977586933776, 0.14132901592908995, 0.14305448896535064, 0.1410781750050576, 0.14747764777554118, 0.1402199633581483, 0.14115302531939486, 0.1491748397441014, 0.15401185024529696, 0.1363709554400133, 0.14256361110702806, 0.13918789055036462, 0.1480231865754594, 0.13157485247306203, 0.14479734975358713, 0.13241208709128524]

    draw1(a, "loss_d_main")