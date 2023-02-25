import imageio
from PIL import Image
import graphviz
from deap import gp


OUTPUT_FILE_DIR = 'test-output/'
OUTPUT_FILE_NAME = 'GEN'


def gif_creat(hof_set):
    """
    A function to creat a gif
    :param hof_set:
    :return:
    """
    gif_name = "NGEN hof.gif"
    dur = 1
    frames = list()

    images_original_list = creat_graph(hof_set)
    images_conved_list = convergence_graph(images_original_list)

    for image_name in images_conved_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=dur)


def convergence_graph(images_original_list):
    """
    Convergence each .png graph to a same size,
    and save it in the dir "adjusted"
    :param images_original_list: the image list that need convergence to same size
    :return images_conved_list: a image name list that convergence to same size
    """
    images_conved_list = list()
    size_list = list()

    for i in range(len(images_original_list)):
        image = Image.open(images_original_list[i])
        size_list.append(image.size)

    max_size = max(size_list)

    for i in range(len(images_original_list)):
        file_name = "GEN " + str(i)
        images_conved_list.append(OUTPUT_FILE_DIR + 'adjusted/' + file_name + '.png')
        image = Image.open(images_original_list[i])
        out = image.resize(max_size)
        out.save(images_conved_list[i])

    return images_conved_list


def creat_graph(hof_set):
    """
    Creat a graph for a set of HallOfFame with a name "GEN *"
     and save it in the dir "best ind"
    :param hof_set: a set of HallOfFame
    :return images_name_list: the saved graph name list
    """
    images_list = list()

    for i in range(len(hof_set)):
        expr = hof_set[i]
        file_name = "GEN " + str(i)
        nodes, edges, labels = gp.graph(expr)
        s_nodes = list(map(str, nodes))
        s_edges = [tuple(map(str, t)) for t in edges]
        s_labels = list(labels.values())
        dot = graphviz.Digraph(comment='NGEN')
        dot.format = 'png'
        for n, l in zip(s_nodes, s_labels):
            dot.node(n, l)
        dot.edges(s_edges)
        dot.attr(size='8,8', ratio='fill')
        # s = file_name + "\nexpr: " + expr + " fitness: " + hof_set[i].fitness.values[0]
        dot.attr(label=file_name)
        dot.attr(fontsize='20')
        images_list.append(OUTPUT_FILE_DIR + "best ind/" + file_name + '.png')
        dot.render(OUTPUT_FILE_DIR + "best ind/" + file_name)

    return images_list
