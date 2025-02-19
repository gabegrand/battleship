import argparse
import json
import os
from io import BytesIO

import numpy as np
import pandas
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportLabImage
from reportlab.platypus import PageBreak
from reportlab.platypus import Paragraph
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle

SHIP_LABELS = {1: "Red Ship", 2: "Green Ship", 3: "Purple Ship", 4: "Orange Ship"}


def create_board_image(board, cell_size=50):
    board_size = len(board)
    width = height = cell_size * (board_size + 1)
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    color_mapping = {
        -1: (200, 200, 200),  # Gray for hidden
        0: (173, 216, 230),  # Light blue for water
        1: (255, 0, 0),  # Red for ship type 1
        2: (0, 255, 0),  # Green for ship type 2
        3: (128, 0, 128),  # Purple for ship type 3
        4: (255, 165, 0),  # Orange for ship type 4
    }

    for i in range(board_size):
        for j in range(board_size):
            x, y = (j + 1) * cell_size, (i + 1) * cell_size
            cell_value = board[i][j]
            draw.rectangle(
                [x, y, x + cell_size, y + cell_size],
                fill=color_mapping[cell_value],
                outline="black",
            )

    # Add labels
    font = ImageFont.load_default()
    for i in range(board_size):
        draw.text(
            (cell_size / 2, (i + 1.5) * cell_size),
            chr(65 + i),
            fill="black",
            anchor="mm",
            font=font,
        )
        draw.text(
            ((i + 1.5) * cell_size, cell_size / 2),
            str(i + 1),
            fill="black",
            anchor="mm",
            font=font,
        )

    return image


def get_ship_tracker(game_board, true_board, ships):
    tracker = []
    for ship in ships:
        ship_seen = np.count_nonzero(game_board == ship)
        ship_true = np.count_nonzero(true_board == ship)
        tracker.append([SHIP_LABELS[ship], ship_seen, ship_true])
    return tracker


def create_first_page(source, game, round, player_data, elements):
    styles = getSampleStyleSheet()

    ids = player_data["id"].tolist()

    elements.append(Paragraph(f"Study Name: {source}", styles["Normal"]))
    elements.append(Paragraph(f"Game ID: {game}", styles["Normal"]))
    elements.append(Paragraph(f"Round ID: {round}", styles["Normal"]))
    elements.append(Spacer(1, 18))

    for id in ids:
        player = player_data[player_data["id"] == id]

        player_prolific_id = player["participantIdentifier"].iloc[0]

        try:
            player_feedback = json.loads(player["exitSurvey"].iloc[0])
        except:
            player_feedback = None

        if player_feedback is not None:
            intro = Paragraph(f"Participant: {player_prolific_id}", styles["Normal"])
            study_rating = Paragraph(
                f'Rated the study {player_feedback["studyLikertValue"]}/7',
                styles["Normal"],
            )
            partner_rating = Paragraph(
                f'Rated their partner {player_feedback["partnerLikertValue"]}/7',
                styles["Normal"],
            )

            study_feedback_text = player_feedback["feedback"]
            if study_feedback_text == "":
                study_feedback_flavor = "Gave no feedback."
            else:
                study_feedback_flavor = (
                    f"Gave the following feedback: '{study_feedback_text}'"
                )

            study_feedback = Paragraph(study_feedback_flavor, styles["Normal"])

            page_elements = [intro, study_rating, partner_rating, study_feedback]

            for element in page_elements:
                elements.append(element)
        else:
            intro = Paragraph(f"Participant: {player_prolific_id}", styles["Normal"])
            none_paragraph = Paragraph(
                f"Did not complete exit survey.", styles["Normal"]
            )
            elements.append(intro)
            elements.append(none_paragraph)

        elements.append(Spacer(1, 12))
    elements.append(PageBreak())


def create_pdf_page(
    game,
    round,
    board,
    true_board,
    message,
    chat_log,
    elements,
    questions_remaining,
    misses,
    hits,
    ship_tracker,
):
    # Convert board image to ReportLab image
    board_image = create_board_image(board)
    temp_image = BytesIO()
    board_image.save(temp_image, format="PNG")
    temp_image.seek(0)
    reportlab_image = ReportLabImage(temp_image, width=3.5 * inch, height=3.5 * inch)

    board_image = create_board_image(true_board)
    temp_image = BytesIO()
    board_image.save(temp_image, format="PNG")
    temp_image.seek(0)
    reportlab_true_image = ReportLabImage(
        temp_image, width=3.5 * inch, height=3.5 * inch
    )

    # Create a paragraph for the latest message
    styles = getSampleStyleSheet()
    message_paragraph = Paragraph(f"Latest Move: {message['text']}", styles["Normal"])

    # Create chat log paragraphs & get no. of questions remaining
    chat_paragraphs = []
    for log_entry in chat_log[-3:]:
        chat_paragraphs.append(Paragraph(log_entry, styles["Normal"]))

    q_remaining_paragraph = Paragraph(
        f"Questions Remaining: {questions_remaining}", styles["Normal"]
    )

    miss_penalty = 0.1
    hit_bonus = 0.2
    starting_bonus = 2
    starting_fraction = 0.25

    performance_bonus = max(
        0,
        (hits * hit_bonus)
        + (starting_bonus * starting_fraction)
        - (misses * miss_penalty),
    )
    misses_paragraph = Paragraph(
        f"Misses: {misses}, Hits: {hits} Performance Bonus: {performance_bonus:.2f}",
        styles["Normal"],
    )

    ship_tracker_paragraph = [Paragraph(f"Ship Tracker:", styles["Normal"])]
    ship_tracker_paragraph += [
        Paragraph(
            f"{ship_tracked[0]}: {ship_tracked[1]}/{ship_tracked[2]}", styles["Normal"]
        )
        for ship_tracked in ship_tracker
    ]

    # Create a table with the images side by side
    image_table = Table(
        [[reportlab_image, reportlab_true_image]], colWidths=[4 * inch, 4 * inch]
    )
    image_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )

    elements.append(image_table)
    elements.append(Spacer(1, 12))
    elements.append(message_paragraph)
    elements.append(Spacer(1, 12))
    elements.extend(chat_paragraphs)
    elements.append(Spacer(1, 12))
    elements.extend(ship_tracker_paragraph)
    elements.append(Spacer(1, 12))
    elements.extend([q_remaining_paragraph, misses_paragraph])
    elements.append(Spacer(1, 175))
    elements.extend([Paragraph(f"Game: {game} Round: {round}")])


def generate_report(source, game, round, index=0):
    elements = []

    all_rounds = pandas.read_csv(f"{source}/round.csv")
    os.makedirs(f"reports_{source}/{game}", exist_ok=True)

    try:
        game_round_data = all_rounds[all_rounds["gameID"] == game]
    except IndexError:
        raise ValueError(f"Invalid Game ID for source {source}")

    try:
        round_data = game_round_data[game_round_data["id"] == round]
    except IndexError:
        raise ValueError(f"Invalid Round ID for game {game} in source {source}")

    # Data for the first page
    all_players = pandas.read_csv(f"{source}/player.csv")
    player_data = all_players[all_players["gameID"] == game]
    player_data = player_data[player_data["ended"] == "game ended"]
    create_first_page(source, game, round, player_data, elements)

    # Load the true final board state
    if type(round_data["trueTiles"].iloc[0]) != float:
        final_board = json.loads(round_data["trueTiles"].iloc[0])
    else:
        return 0

    # Create initial board (all -1s)
    initial_board = np.full((len(final_board), len(final_board)), -1)
    game_board = initial_board

    # Parse JSON messages
    messages = json.loads(round_data["messages"].iloc[0])

    # Load spotter ratings
    ratings_list = json.loads(round_data["spotterRatings"].iloc[0])

    if ratings_list != []:
        ratings = {rating[0]: rating[1] for rating in ratings_list}
    else:
        ratings = None

    # Initialize chat log and content list
    chat_log = []

    # Process messages
    questions_remaining = args.total_questions
    for message in messages:
        question_rating = None

        # Count Qs Remaining, Handle Ratings
        if message["type"] == "question" and message["text"] != "(question skipped)":
            questions_remaining -= 1
            if (ratings is not None) and (message["text"] in ratings.keys()):
                question_rating = ratings[message["text"]]
            else:
                question_rating = None

        if message["type"] != "decision":
            if question_rating is None:
                chat_log.append(
                    f"({message['time']} ms) {message['type'].capitalize()}: {message['text']}"
                )
            else:
                chat_log.append(
                    f"({message['time']} ms) {message['type'].capitalize()}: {message['text']} | Spotter Rating: {question_rating}"
                )

        # Handle Move, Update Board, Generate Page
        if message["type"] == "move":
            if message["text"] != "(firing timed out)":
                row = ord(message["text"][0]) - ord("A")
                col = int(message["text"][1:]) - 1
                game_board[row][col] = final_board[row][col]

            # Additional Tidbits
            misses = np.count_nonzero(game_board == 0)
            hits = np.count_nonzero(game_board > 0)
            ship_tracker = get_ship_tracker(
                game_board, np.array(final_board), args.ships
            )

            create_pdf_page(
                game,
                round,
                game_board,
                final_board,
                message,
                chat_log,
                elements,
                questions_remaining,
                misses,
                hits,
                ship_tracker,
            )
        if message["type"] == "answer":
            misses = np.count_nonzero(game_board == 0)
            hits = np.count_nonzero(game_board > 0)
            ship_tracker = get_ship_tracker(
                game_board, np.array(final_board), args.ships
            )
            create_pdf_page(
                game,
                round,
                game_board,
                final_board,
                message,
                chat_log,
                elements,
                questions_remaining,
                misses,
                hits,
                ship_tracker,
            )

    # Create the PDF
    doc = SimpleDocTemplate(
        f"reports_{source}/{game}/report_{index}_{round}.pdf", pagesize=letter
    )
    doc.build(elements)


def get_all_games(folder):
    all_games = pandas.read_csv(f"{folder}/game.csv")
    game_list = all_games[all_games["actualPlayerCount"] == 2]["id"].tolist()
    return game_list


def list_games(folder):
    game_list = get_all_games(folder)
    print(f"Game IDs for games with 2 players in {folder}:")
    for game in game_list:
        print(f"- {game}")
    print(f"(Total: {len(game_list)} games)")
    exit(code=0)


def get_all_rounds(folder, game):
    all_rounds = pandas.read_csv(f"{folder}/round.csv")
    rounds_and_indices = all_rounds[all_rounds["gameID"] == game][
        ["id", "index"]
    ].sort_values("index")
    round_list = rounds_and_indices["id"].tolist()
    indices = rounds_and_indices["index"].tolist()
    return round_list, indices


def list_rounds(folder, game):
    round_list, indices = get_all_rounds(folder, game)
    print(f"Round IDs for game {game} in {folder}:")
    for index, round in enumerate(round_list):
        print(f"{indices[index]}. {round}")
    print(f"(Total: {len(round_list)} rounds)")
    exit(code=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--game", type=str, required=False)
    parser.add_argument("--round", type=str, required=False)
    parser.add_argument("--total_questions", type=str, required=False, default=15)
    parser.add_argument("--total_games", type=str, required=False, default=6)
    parser.add_argument("--ships", type=list, required=False, default=[1, 2, 3, 4])
    parser.add_argument(
        "--list_games", action="store_const", const=True, required=False, default=False
    )
    parser.add_argument(
        "--list_rounds", action="store_const", const=True, required=False, default=False
    )
    parser.add_argument(
        "--generate_report",
        action="store_const",
        const=True,
        required=False,
        default=False,
    )

    args = parser.parse_args()

    if args.list_games:
        print("Listing games...")
        list_games(args.source)

    if args.list_rounds:
        print("Listing rounds...")
        list_rounds(args.source, args.game)

    if args.generate_report:
        os.makedirs(f"reports_{args.source}", exist_ok=True)
        if args.game is not None:
            games = args.game
        else:
            games = get_all_games(args.source)

        if args.round is not None:
            rounds = args.round
            generate_report(args.source, games, rounds)
        else:
            if isinstance(games, list):
                for game_index, game in enumerate(games):
                    rounds, indices = get_all_rounds(args.source, game)
                    # makes sure reports are only generated for full games (n rounds)
                    #'>' is intentional: the tutorial game is counted as a round
                    # so '=>' would also count games with n-1 rounds instead of n
                    if len(rounds) > int(args.total_games) or args.game is not None:
                        for round_index, round in enumerate(rounds):
                            if round_index == 0:
                                continue  # skips the tutorial 'round'
                            print(
                                f"Generating report for {args.source}: game {game_index+1}/{len(games)} ({game}), round {round_index + 1}/{len(rounds)}",
                                end="\r",
                            )
                            generate_report(
                                args.source, game, round, indices[round_index]
                            )
                print("All reports successfully generated!" + " " * 80)
