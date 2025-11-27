import xml.etree.ElementTree as ET

def parse_xml(xml_text, source_date, all_matches):
    try:
        root = ET.fromstring(xml_text)
    except:
        return  # invalid XML

    for category in root.findall("category"):
        category_name = category.get("name", "")
        category_date = category.get("date", "")
        category_id = category.get("id", "")

        for match in category.findall("match"):
            match_id = match.get("id", "")
            match_date = match.get("date", "")
            match_time = match.get("time", "")
            match_status = match.get("status", "")

            localteam = match.find("localteam")
            awayteam = match.find("awayteam")

            localteam_name = localteam.get("name", "") if localteam is not None else ""
            awayteam_name = awayteam.get("name", "") if awayteam is not None else ""

            local_winner = localteam.get("winner", "") if localteam is not None else ""
            away_winner = awayteam.get("winner", "") if awayteam is not None else ""

            local_id = localteam.get("id", "") if localteam is not None else ""
            away_id = awayteam.get("id", "") if awayteam is not None else ""

            win_by = match.find("win_result/won_by")
            win_type = win_by.get("type", "") if win_by is not None else ""
            win_round = win_by.get("round", "") if win_by is not None else ""
            win_minute = win_by.get("minute", "") if win_by is not None else ""

            ko_tag = win_by.find("ko") if win_by is not None else None
            sub_tag = win_by.find("sub") if win_by is not None else None
            points_tag = win_by.find("points") if win_by is not None else None

            won_by_ko_type = ko_tag.get("type", "") if ko_tag is not None else ""
            won_by_ko_target = ko_tag.get("target", "") if ko_tag is not None else ""

            win_sub_type = sub_tag.get("type", "") if sub_tag is not None else ""
            win_points_score = points_tag.get("score", "") if points_tag is not None else ""

            stats = match.find("stats")

            # Local stats
            local_stats = stats.find("localteam") if stats is not None else None
            lt_strikes_total = local_stats.find("strikes_total") if local_stats is not None else None
            lt_strikes_power = local_stats.find("strikes_power") if local_stats is not None else None
            lt_takedowns = local_stats.find("takedowns") if local_stats is not None else None
            lt_submissions = local_stats.find("submissions") if local_stats is not None else None
            lt_control_time = local_stats.find("control_time") if local_stats is not None else None
            lt_knockdowns = local_stats.find("knockdowns") if local_stats is not None else None

            # Away stats
            away_stats = stats.find("awayteam") if stats is not None else None
            aw_strikes_total = away_stats.find("strikes_total") if away_stats is not None else None
            aw_strikes_power = away_stats.find("strikes_power") if away_stats is not None else None
            aw_takedowns = away_stats.find("takedowns") if away_stats is not None else None
            aw_submissions = away_stats.find("submissions") if away_stats is not None else None
            aw_control_time = away_stats.find("control_time") if away_stats is not None else None
            aw_knockdowns = away_stats.find("knockdowns") if away_stats is not None else None

            all_matches.append({
                "source_date": source_date,

                "category": category_name,
                "category_date": category_date,
                "category_id": category_id,

                "match_id": match_id,
                "date": match_date,
                "time": match_time,
                "status": match_status,

                "localteam_name": localteam_name,
                "local_id": local_id,
                "awayteam_name": awayteam_name,
                "away_id": away_id,
                "local_winner": local_winner,
                "away_winner": away_winner,

                "win_type": win_type,
                "win_round": win_round,
                "win_minute": win_minute,
                "won_by_ko_type": won_by_ko_type,
                "won_by_ko_target": won_by_ko_target,
                "win_sub_type": win_sub_type,
                "win_points_score": win_points_score,

                "local_strikes_total_head": lt_strikes_total.get("head", "") if lt_strikes_total is not None else "",
                "local_strikes_total_body": lt_strikes_total.get("body", "") if lt_strikes_total is not None else "",
                "local_strikes_total_leg": lt_strikes_total.get("legs", "") if lt_strikes_total is not None else "",

                "local_strikes_power_head": lt_strikes_power.get("head", "") if lt_strikes_power is not None else "",
                "local_strikes_power_body": lt_strikes_power.get("body", "") if lt_strikes_power is not None else "",
                "local_strikes_power_leg": lt_strikes_power.get("legs", "") if lt_strikes_power is not None else "",

                "local_takedowns_att": lt_takedowns.get("att", "") if lt_takedowns is not None else "",
                "local_takedowns_landed": lt_takedowns.get("landed", "") if lt_takedowns is not None else "",

                "local_submissions": lt_submissions.get("total", "") if lt_submissions is not None else "",
                "local_control_time": lt_control_time.get("total", "") if lt_control_time is not None else "",
                "local_knockdowns": lt_knockdowns.get("total", "") if lt_knockdowns is not None else "",

                "away_strikes_total_head": aw_strikes_total.get("head", "") if aw_strikes_total is not None else "",
                "away_strikes_total_body": aw_strikes_total.get("body", "") if aw_strikes_total is not None else "",
                "away_strikes_total_leg": aw_strikes_total.get("legs", "") if aw_strikes_total is not None else "",

                "away_strikes_power_head": aw_strikes_power.get("head", "") if aw_strikes_power is not None else "",
                "away_strikes_power_body": aw_strikes_power.get("body", "") if aw_strikes_power is not None else "",
                "away_strikes_power_leg": aw_strikes_power.get("legs", "") if aw_strikes_power is not None else "",

                "away_takedowns_att": aw_takedowns.get("att", "") if aw_takedowns is not None else "",
                "away_takedowns_landed": aw_takedowns.get("landed", "") if aw_takedowns is not None else "",

                "away_submissions": aw_submissions.get("total", "") if aw_submissions is not None else "",
                "away_control_time": aw_control_time.get("total", "") if aw_control_time is not None else "",
                "away_knockdowns": aw_knockdowns.get("total", "") if aw_knockdowns is not None else "",
            })
