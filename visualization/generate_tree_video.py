import numpy as np
from cairosvg import svg2png
from collections import defaultdict

color_prediction = "#fd9731"  ##72e8cd"  # "#15afda"
color_gt = "#15da78"
color_level =   "#ee538b"
color_default = "#ffffff"
color_background = "#2f3337"


def main():
    a = create_figure('finegym', None, ground_truth=200, prediction=None, percentage_pred=None, num_dots=3)
    # create_figure('finegym', 3, ground_truth=7, prediction=19)
    # a = create_figure('finegym', 2, ground_truth=7, prediction=6)
    a.save_image('/proj/vondrick/didac/results/prova.png')


def create_figure(dataset, selected_level, ground_truth, prediction, percentage_pred, num_dots):
    """
    Ground truth is a list of a label per level if dataset is hollywood2, else a number representing the id of the last level (total id)
    Prediction is the same but in hollywood2 the list only needs to go up to selected_level, and in finegym the id refers to the selected level
    """
    global w, h, size_text, vertical_sep, horizontal_sep, h_margin, v_margin, arrow_start_x, arrow_end_x, \
        arrow_start_y, arrow_end_y

    scale = 1.9 if dataset == 'hollywood2' else 1.5
    w = 600/scale
    h = 720/scale
    size_text = 10
    size_title = 15
    vertical_sep = np.array([30, 25, 20, 15]) if dataset == 'hollywood2' else np.array([28, 23, 18, 15])
    horizontal_sep = 60 if dataset == 'hollywood2' else 30
    h_margin = 5
    v_margin = 70
    title_height = 40
    arrow_start_x = horizontal_sep / 2
    arrow_end_x = 2
    arrow_start_y = 5 if dataset == 'hollywood2' else 3
    arrow_end_y = size_text / 3

    if dataset == 'hollywood2':
        assert selected_level is None or 1 <= selected_level <= 2
        level_dict = {
            'Hollywood2 (Root)':
                {
                    'Car Interaction': {'Drive Car': None, 'Get Out Car': None},
                    'Handle Tools': {'Answer Phone': None, 'Eat': None},
                    'Interact with Human': {'Hug Person': None, 'Kiss': None, 'Hand Shake': None},
                    'Fast Human Action': {'Fight Person': None, 'Run': None},
                    'Body Motion': {'Sit Down': None, 'Sit Up': None, 'Stand Up': None}
                },
        }

        # {'HugPerson': '0',
        #  'StandUp': '1',
        #  'Kiss': '2',
        #  'FightPerson': '3',
        #  'GetOutCar': '4',
        #  'Run': '5',
        #  'SitUp': '6',
        #  'DriveCar': '7',
        #  'SitDown': '8',
        #  'AnswerPhone': '9',
        #  'Eat': '10',
        #  'HandShake': '11',
        #  'Car': '12',
        #  'Tools': '13',
        #  'HumanInteraction': '14',
        #  'FastHumanAction': '15',
        #  'BodyMotion': '16',
        #  'Root': '17'}

        dict_to_index = {0: [2, 0], 1: [4, 2], 2: [2, 1], 3: [3, 0], 4: [0, 1], 5: [3, 1], 6: [4, 1], 7: [0, 0],
                         8: [4, 0], 9: [1, 0], 10: [1, 1], 11: [2, 2], 12: [0], 13: [1], 14: [2], 15: [3], 16: [4]}
        if selected_level is not None:
            prediction = dict_to_index[prediction[0 if selected_level == 2 else 1]]
            ground_truth = dict_to_index[ground_truth[0]]
    else:  # finegym
        assert selected_level is None or 1 <= selected_level <= 3
        path_finegym_categories = '/proj/vondrick/datasets/FineGym/categories/gym288_categories.txt'
        path_finegym_set_cat = '/proj/vondrick/datasets/FineGym/categories/set_categories.txt'
        # path_finegym_categories = '/Users/didac/proj/datasets/FineGym/categories/gym288_categories.txt'
        # path_finegym_set_cat = '/Users/didac/proj/datasets/FineGym/categories/set_categories.txt'

        level_dict = defaultdict(lambda: defaultdict(dict))

        dict_index_to_hier_low = {}
        dict_index_to_hier_mid = {}

        with open(path_finegym_categories, 'r') as f:
            categories = f.readlines()
        with open(path_finegym_set_cat, 'r') as f:
            set_categories = f.readlines()
        dict_set_categories = {}
        for line in set_categories:
            line = line.split(";")
            dict_set_categories[int(line[0][-2:])] = line[1][1:]
        list_set_categories = list(dict_set_categories.keys())
        dict_top_categories = {"VT": "Vault", "FX": "Floor Exercise", "UB": "Uneven Bars", "BB": "Balance Beam"}
        for i, line in enumerate(categories):
            low_class = int(line[7:11])
            mid_class_num = int(line[17:20])
            mid_class = dict_set_categories[mid_class_num]
            top_class = dict_top_categories[line[36:38]]
            text_low = line[40:]

            level_dict[top_class][mid_class][f'{low_class}: {text_low}'] = None
            dict_index_to_hier_low[low_class] = [len(level_dict) - 1, len(level_dict[top_class]) - 1,
                                                 len(level_dict[top_class][mid_class]) - 1]
            dict_index_to_hier_mid[list_set_categories.index(mid_class_num)] = [len(level_dict) - 1,
                                                                                len(level_dict[top_class]) - 1]

        level_dict = {'FineGym': level_dict}
        ground_truth = dict_index_to_hier_low[ground_truth]
        if prediction is not None:
            if selected_level == 1:
                prediction = [prediction]
            elif selected_level == 2:
                prediction = dict_index_to_hier_mid[prediction]
            else:
                prediction = dict_index_to_hier_low[prediction]

    elements = []
    elements.append((Rectangle(0, 0, w, h, color_background, color_background), -2))
    rows_printed = np.array([0, 0, 0, 0])
    title = 'Future action prediction' if dataset == 'finegym' else 'Early action recognition'
    elements.append((Text(x=h_margin, y=title_height, text=title, size=size_title), 0))
    create_hierarchy(level_dict, 0, elements, rows_printed, 0, selected_level, ground_truth, prediction,
                     percentage_pred, num_dots=num_dots)

    image = Image(w=w, h=h, scale=scale)
    for element, priority in elements:
        image.add_element(element, priority)

    if selected_level is not None:
        # vertical line with selected level
        arrow = Arrow(x=h_margin + horizontal_sep * selected_level, y=2, color_fill=color_level, color_stroke=color_level,
                      stroke_width=3)
        arrow.add_vertical_line(end_point=v_margin - size_text, is_last=True)
        # image.add_element(arrow, priority=-1)

        arrow = Arrow(x=h_margin + horizontal_sep * selected_level, y=v_margin, color_stroke=color_level, opacity=0.3,
                      stroke_width=3)
        arrow.add_vertical_line(end_point=h)
        # image.add_element(arrow, priority=-1)

    return image


def create_hierarchy(level_dict, level, elements, rows_printed, initial_height, selected_level, ground_truth,
                     prediction, percentage_pred, gt_previous=True, pred_previous=True, dots_previous=False,
                     num_dots=3):
    num_before_dots = 3
    height_text = None
    for i, (level_k, level_v) in enumerate(level_dict.items()):
        dots = False
        gt_previous_, pred_previous_ = gt_previous, pred_previous
        if level >= 2 and ((not pred_previous_) or selected_level < level) and (not gt_previous):
            if i==0 and not dots_previous:
                dots = True
            else:
                continue
        # If gt_previous or pred_previous but far from current
        if (not dots) and level == 3 and prediction is not None and \
                ((i < prediction[level-1] - num_before_dots or i > prediction[level-1] + num_before_dots) and
                 (i < ground_truth[level-1] - num_before_dots or i > ground_truth[level-1] + num_before_dots)):
            if (pred_previous_ and (i == prediction[level-1] - (num_before_dots+1) or i == prediction[level-1] + num_before_dots+1)) or\
                    (gt_previous_ and (i == ground_truth[level-1] - (num_before_dots+1) or i == ground_truth[level-1] + num_before_dots+1)):
                dots = True
            else:
                continue

        if level > 0 or i > 0:  # no need for spacing on top of the first title
            rows_printed[level] += 1  # on top because we want the spacing to be the one on top of the title

        priority = 0
        color = color_default
        # if gt_previous_ and level > 0:
        #     if i == ground_truth[level - 1]:
        #         priority = 2
        #         color = color_gt
        #     else:
        #         gt_previous_ = False
        gt_previous_ = False

        if pred_previous_ and level > 0:
            if prediction is None:
                pred_previous_ = False
            elif selected_level >= level and i == prediction[level - 1]:
                priority = 3
                color = color_prediction
            else:
                pred_previous_ = False

        height_text_old = height_text
        height_text = v_margin + (rows_printed * vertical_sep).sum()
        max_text = 50 if not color == color_prediction else 43
        text = level_k if len(level_k) < max_text else level_k[:max_text-3] + "..."
        text = text.replace("_", ' ').replace('\n', '')
        if level == selected_level and color == color_prediction:
            text = f"{text} ({int(percentage_pred*100):d} %)"
        if dots:
            text = ". "*num_dots
        elements.append((Text(x=h_margin + horizontal_sep * level, y=height_text, text=text, size=size_text,
                              color=color), priority))

        if level > 0:
            if height_text_old is None or color != color_default:
                init_arrow_vertical = initial_height + arrow_start_y
            else:
                init_arrow_vertical = height_text_old - arrow_end_y
            arrow = Arrow(x=h_margin + horizontal_sep * (level - 1) + arrow_start_x, y=init_arrow_vertical,
                          color_fill=color, color_stroke=color)
            arrow.add_vertical_line(end_point=v_margin + (rows_printed * vertical_sep).sum() - arrow_end_y)
            arrow.add_horizontal_line(end_point=h_margin + horizontal_sep * level - arrow_end_x, is_last=True)
            elements.append((arrow, priority))

        if level_v is not None:
            create_hierarchy(level_v, level + 1, elements, rows_printed, height_text, selected_level, ground_truth,
                             prediction, percentage_pred, gt_previous_, pred_previous_, dots_previous=dots,
                             num_dots=num_dots)


class Image():
    def __init__(self, w, h, color="#ffffff", scale=1.):
        self.w = w
        self.h = h
        self.color = color
        self.ending = '</svg>'
        self.image_elements = defaultdict(list)
        self.view_box = [0, 0, self.w, self.h]
        self.scale = scale

    def save_image(self, path='image_svg.png'):
        priorities = sorted(self.image_elements.keys(), reverse=True)
        image_elements = []
        for priority in priorities:
            image_elements = self.image_elements[priority] + image_elements
        image_string = self.return_header() + ''.join(
            [element.return_string() for element in image_elements]) + self.ending
        svg2png(bytestring=image_string, write_to=path, scale=self.scale)

    # with open(path, 'w') as f:
    # 	f.write(image_string)

    def add_element(self, element, priority=0):
        self.image_elements[priority].append(element)

    def return_header(self):
        # Not sure why creating the background color with background-color as a style property does not work
        return f'<svg width="{self.w}px" height="{self.h}px" viewBox="{" ".join([str(v) for v in self.view_box])}"><rect width="100%" height="100%" fill="{self.color}"/>'


class Text():
    def __init__(self, x, y, text, color=color_default, size=12, font="Helvetica"):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.size = size
        self.font = font
        self.text_anchor = "left"  # alternative: middle. What part of the text the positions x, y refer to

    def return_string(self):
        font_weight = "normal" if self.color == color_default else "bold"
        size = self.size if self.color == color_default else int(self.size*1.3)
        s = f'<text x="{self.x}" y="{self.y}" fill="{self.color}" font-family="{self.font}" font-size="{size}px" text-anchor="{self.text_anchor}" font-weight="{font_weight}">{self.text}</text>'
        return s


class Rectangle():
    def __init__(self, x, y, w, h, color_fill="none", color_stroke=color_default):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color_fill = color_fill
        self.color_stroke = color_stroke

    def return_string(self):
        s = f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" fill="{self.color_fill}" stroke="{self.color_stroke}"/>'
        return s


class Arrow():
    """
    Each of the commands is instantiated (for example, creating a class, naming and locating it) by a specific letter. For instance, let's move to the x and y coordinates (10, 10).
    The "Move to" command is called with the letter M. When the parser runs into this letter, it knows it needs to move to a point. So, to move to (10,10) the command to use would
    be M 10 10. After that, the parser begins reading for the next command.
    All of the commands also come in two variants. An uppercase letter specifies absolute coordinates on the page, and a lowercase letter specifies relative coordinates (e.g., move
    10px up and 7px to the left from the last point).
    M (m): move to (x, y)
    L (l): line to (x, y)
    H (h): horizontal line (x)
    V (v): vertical line (y)
    Z (same as z): close the polygon
    """

    def __init__(self, x, y, color_fill=color_default, color_stroke=color_default, add_arrow_pointer=True,
                 stroke_width=1, opacity=1.):
        self.x = x
        self.y = y
        self.color_fill = color_fill  # only for the pointer
        self.color_stroke = color_stroke
        self.pointer_arrow = None
        self.add_arrow_pointer = add_arrow_pointer
        self.current_position = (x, y)
        self.commands = []
        self.stroke_width = stroke_width if color_stroke == color_default else stroke_width * 3
        self.opacity = opacity

    def add_line(self, end_point=None, inc=None, is_last=False):
        assert end_point is not None or inc is not None
        if end_point:
            # self.commands.append(f"L {end_point[0]} {end_point[1]}")
            inc = (end_point[0] - self.current_position[0], end_point[1] - self.current_position[1])
        else:
            end_point = (self.current_position[0] + inc[0], self.current_position[1] + inc[1])
        if is_last and self.add_arrow_pointer:
            self.commands.append(
                f"l {inc[0] - self.stroke_width * inc[0] / np.sqrt(inc[0] ** 2 + inc[1] ** 2)} {inc[1] - self.stroke_width * inc[1] / np.sqrt(inc[0] ** 2 + inc[1] ** 2)}")
        else:
            self.commands.append(f"l {inc[0]} {inc[1]}")
        angle = np.pi / 2 if inc[0] == 0 else np.arctan(inc[1] / inc[0])
        if inc[0] < 0:
            angle += np.pi
        self.current_position = end_point
        if is_last and self.add_arrow_pointer:
            self.pointer_arrow = PointerArrow(self.current_position[0], self.current_position[1], angle,
                                              color_fill=self.color_fill,
                                              color_stroke=self.color_stroke, opacity=self.opacity,
                                              size=np.sqrt(float(self.stroke_width)) * 5)

    def add_horizontal_line(self, end_point=None, inc=None, is_last=False):
        assert end_point is not None or inc is not None
        end_point = None if end_point is None else (end_point, self.current_position[1])
        inc = None if inc is None else (inc, 0)
        self.add_line(end_point=end_point, inc=inc, is_last=is_last)

    def add_vertical_line(self, end_point=None, inc=None, is_last=False):
        assert end_point is not None or inc is not None
        end_point = None if end_point is None else (self.current_position[0], end_point)
        inc = None if inc is None else (0, inc)
        self.add_line(end_point=end_point, inc=inc, is_last=is_last)

    def return_string(self):
        s = f'<path d="M {self.x} {self.y} {" ".join(self.commands)}" fill="none" stroke="{self.color_stroke}" stroke-miterlimit="10" ' + \
            f'opacity="{self.opacity}" stroke-width="{self.stroke_width}"/>'
        if self.add_arrow_pointer and self.pointer_arrow is not None:
            s += self.pointer_arrow.return_string()
        return s


class PointerArrow():
    def __init__(self, x, y, angle=0, size=5, color_fill=color_default, color_stroke=color_default, flatness=0.7,
                 opacity="1"):
        self.x = x  # the pointer of the arrow
        self.y = y
        self.angle = angle
        self.size = size  # size of the side of the arrow
        self.color_fill = color_fill
        self.color_stroke = color_stroke
        self.flatness = flatness  # 0 is a flat arrow, 1 is an arrow without filling
        self.opacity = opacity

        angle_top = self.angle - np.pi / 8
        self.point_top = (self.x - self.size * np.cos(angle_top), self.y - self.size * np.sin(angle_top))
        angle_mid = self.angle
        self.point_mid = (
        self.x - self.flatness * self.size * np.cos(angle_mid), self.y - self.flatness * self.size * np.sin(angle_mid))
        angle_bottom = self.angle + np.pi / 8
        self.point_bottom = (self.x - self.size * np.cos(angle_bottom), self.y - self.size * np.sin(angle_bottom))

    def return_string(self):
        s = f'<path d="M {self.x} {self.y} L {self.point_top[0]} {self.point_top[1]} L {self.point_mid[0]} {self.point_mid[1]} L {self.point_bottom[0]} {self.point_bottom[1]} Z" ' + \
            f'fill="{self.color_fill}" stroke="{self.color_stroke}" stroke-miterlimit="10" opacity="{self.opacity}"/>'
        return s


if __name__ == '__main__':
    main()


