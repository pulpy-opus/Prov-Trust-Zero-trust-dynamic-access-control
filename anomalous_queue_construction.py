import logging
import json
import re

import torch

from Prov_Trust_utils import *
from config import *

# Setting for logging
logger = logging.getLogger("anomalous_queue_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'anomalous_queue.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# IP behavior process logger
ip_behavior_process_logger = logging.getLogger("ip_behavior_process_logger")
ip_behavior_process_logger.setLevel(logging.INFO)
ip_behavior_process_file_handler = logging.FileHandler(artifact_dir + 'ip_behavior_process.log')
ip_behavior_process_file_handler.setLevel(logging.INFO)
ip_behavior_process_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
ip_behavior_process_file_handler.setFormatter(ip_behavior_process_formatter)
if not ip_behavior_process_logger.hasHandlers():
    ip_behavior_process_logger.addHandler(ip_behavior_process_file_handler)


def edge_stats(edges_dict):
    """Compute statistics for edges."""
    stats = {}
    for edge_tuple, loss_list in edges_dict.items():
        if loss_list:
            stats[str(edge_tuple)] = {
                'count': len(loss_list),
                'mean_loss': float(np.mean(loss_list)),
                'max_loss': float(max(loss_list)),
                'min_loss': float(min(loss_list))
            }
    return stats


def cal_anomaly_loss(loss_list, edge_list):
    if len(loss_list) != len(edge_list):
        print("error!")
        return 0, 0, set(), set(), 0
    count = 0
    loss_sum = 0
    loss_std = std(loss_list)
    loss_mean = mean(loss_list)
    edge_set = set()
    node_set = set()

    thr = loss_mean + 2 * loss_std

    logger.info(f"thr:{thr}")

    for i in range(len(loss_list)):
        if loss_list[i] > thr:
            count += 1
            src_node = edge_list[i][0]
            dst_node = edge_list[i][1]
            loss_sum += loss_list[i]

            node_set.add(src_node)
            node_set.add(dst_node)
            edge_set.add(edge_list[i][0] + edge_list[i][1])

    if count == 0:
        return count, 0, node_set, edge_set, thr
    return count, loss_sum / count, node_set, edge_set, thr

def compute_IDF():
    node_IDF = {}

    file_list = []
    file_path = artifact_dir + "graph_4_3/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    file_path = artifact_dir + "graph_4_4/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    file_path = artifact_dir + "graph_4_5/"
    file_l = os.listdir(file_path)
    for i in file_l:
        file_list.append(file_path + i)

    node_set = {}
    for f_path in tqdm(file_list):
        f = open(f_path)
        for line in f:
            l = line.strip()
            jdata = eval(l)
            if jdata['loss'] > 0:
                if 'netflow' not in str(jdata['srcmsg']):
                    if str(jdata['srcmsg']) not in node_set.keys():
                        node_set[str(jdata['srcmsg'])] = {f_path}
                    else:
                        node_set[str(jdata['srcmsg'])].add(f_path)
                if 'netflow' not in str(jdata['dstmsg']):
                    if str(jdata['dstmsg']) not in node_set.keys():
                        node_set[str(jdata['dstmsg'])] = {f_path}
                    else:
                        node_set[str(jdata['dstmsg'])].add(f_path)
    for n in node_set:
        include_count = len(node_set[n])
        IDF = math.log(len(file_list) / (include_count + 1))
        node_IDF[n] = IDF

    torch.save(node_IDF, artifact_dir + "node_IDF")
    logger.info("IDF weight calculate complete!")
    return node_IDF, file_list

# Measure the relationship between two time windows, if the returned value
# is not 0, it means there are suspicious nodes in both time windows.
def cal_set_rel(s1, s2, node_IDF, tw_list):
    def is_include_key_word(s):
        # The following common nodes don't exist in the training/validation data, but
        # will have the influences to the construction of anomalous queue (i.e. noise).
        # These nodes frequently exist in the testing data but don't contribute much to
        # the detection (including temporary files or files with random name).
        # Assume the IDF can keep being updated with the new time windows, these
        # common nodes can be filtered out.
        keywords = [
            'netflow',
            '/home/george/Drafts',
            'usr',
            'proc',
            'var',
            'cadet',
            '/var/log/debug.log',
            '/var/log/cron',
            '/home/charles/Drafts',
            '/etc/ssl/cert.pem',
            '/tmp/.31.3022e',
        ]
        flag = False
        for i in keywords:
            if i in s:
                flag = True
        return flag

    new_s = s1 & s2
    count = 0
    for i in new_s:
        if is_include_key_word(i) is True:
            node_IDF[i] = math.log(len(tw_list) / (1 + len(tw_list)))

        if i in node_IDF.keys():
            IDF = node_IDF[i]
        else:
            # Assign a high IDF for those nodes which are neither in training/validation
            # sets nor excluded node list above.
            IDF = math.log(len(tw_list) / (1))

        # Compare the IDF with a rareness threshold α
        if IDF > (math.log(len(tw_list) * 0.9)):
            logger.info(f"node:{i}, IDF:{IDF}")
            count += 1
    return count

def anomalous_queue_construction(node_IDF, tw_list, graph_dir_path):
    history_list = []
    current_tw = {}

    # Create netflow_ip_logs directory
    netflow_log_dir = artifact_dir + "netflow_ip_logs"
    if not os.path.exists(netflow_log_dir):
        os.makedirs(netflow_log_dir)

    # Create ip_behavior_logs directory
    iplog_dir = artifact_dir + "ip_behavior_logs"
    if not os.path.exists(iplog_dir):
        os.makedirs(iplog_dir)

    file_l = os.listdir(graph_dir_path)
    index_count = 0
    for f_path in sorted(file_l):
        logger.info("**************************************************")
        logger.info(f"Time window: {f_path}")

        # Generate time interval identifier
        time_interval = f_path.replace('.txt', '').replace('/', '_')

        f = open(f"{graph_dir_path}/{f_path}")
        edge_loss_list = []
        edge_list = []
        netflow_ip_port_dict = {}
        logger.info(f'Time window index: {index_count}')

        # Figure out which nodes are anomalous in this time window
        for line in f:
            l = line.strip()
            jdata = eval(l)
            edge_loss_list.append(jdata['loss'])
            edge_list.append([str(jdata['srcmsg']), str(jdata['dstmsg'])])

            # Collect netflow IP and port info
            try:
                srcmsg_dict = eval(str(jdata['srcmsg'])) if str(jdata['srcmsg']).startswith('{') else {}
            except:
                srcmsg_dict = {}
            if 'netflow' in srcmsg_dict:
                match = re.match(r"([0-9]{1,3}(?:\.[0-9]{1,3}){3}):(\d+)", srcmsg_dict['netflow'])
                if match:
                    ip = match.group(1)
                    port = match.group(2)
                    if ip not in netflow_ip_port_dict:
                        netflow_ip_port_dict[ip] = set()
                    netflow_ip_port_dict[ip].add(port)

        # Save netflow IP/port log for this time window
        netflow_log_path = os.path.join(netflow_log_dir, f"netflow_ip_log_{time_interval}.json")
        netflow_ip_port_dict_serializable = {ip: list(ports) for ip, ports in netflow_ip_port_dict.items()}
        with open(netflow_log_path, "w") as f_netlog:
            json.dump(netflow_ip_port_dict_serializable, f_netlog, ensure_ascii=False, indent=2)

        count, loss_avg, node_set, edge_set, thr = cal_anomaly_loss(edge_loss_list, edge_list)
        current_tw['name'] = f_path
        current_tw['loss'] = loss_avg
        current_tw['index'] = index_count
        current_tw['nodeset'] = node_set

        # Process IP behavior logs: label anomalies and compute stats
        iplog_path = os.path.join(iplog_dir, f"ip_behavior_log_{time_interval}.json")
        abnormal_ip_log = {}
        ip_stats = {}

        if os.path.exists(iplog_path):
            with open(iplog_path, "r") as f_iplog:
                ip_behavior_log = json.load(f_iplog)
            ip_count = sum(len(logs) for logs in ip_behavior_log.values())

            # Log processing info
            ip_behavior_process_logger.info(
                f"Time window: {time_interval}, IP behavior count: {ip_count}, thr: {thr}, matched IP behavior log: {iplog_path}")

            for ip, logs in ip_behavior_log.items():
                # Count anomalous/normal behaviors and edge types for this IP in this time window
                abnormal_entries = []
                normal_entries = []
                abnormal_edges = {}
                normal_edges = {}

                for entry in logs:
                    edge_tuple = (entry.get('srcnode', ''), entry.get('dstnode', ''))
                    loss_val = entry.get('loss', 0)
                    # Determine if anomalous: edge in edge_set or loss exceeds threshold
                    edge_key = str(entry.get('srcnode', '')) + str(entry.get('dstnode', ''))
                    if edge_key in edge_set or loss_val > thr:
                        abnormal_entries.append(entry)
                        if edge_tuple not in abnormal_edges:
                            abnormal_edges[edge_tuple] = []
                        abnormal_edges[edge_tuple].append(loss_val)
                    else:
                        normal_entries.append(entry)
                        if edge_tuple not in normal_edges:
                            normal_edges[edge_tuple] = []
                        normal_edges[edge_tuple].append(loss_val)

                abnormal_edge_stats = edge_stats(abnormal_edges)
                normal_edge_stats = edge_stats(normal_edges)

                if abnormal_entries or normal_entries:
                    abnormal_ip_log[ip] = {
                        'time_window': time_interval,
                        'abnormal_count': len(abnormal_entries),
                        'normal_count': len(normal_entries),
                        'abnormal_edges': [str(e) for e in list(abnormal_edges.keys())],
                        'normal_edges': [str(e) for e in list(normal_edges.keys())],
                        'abnormal_edge_types': list(set([e.get('edge_type') for e in abnormal_entries if 'edge_type' in e])),
                        'abnormal_edge_stats': abnormal_edge_stats,
                        'normal_edge_stats': normal_edge_stats,
                        'entries': abnormal_entries,
                        'normal_entries': normal_entries
                    }
                    ip_stats[ip] = {
                        'time_window': time_interval,
                        'abnormal_count': len(abnormal_entries),
                        'normal_count': len(normal_entries),
                        'abnormal_edges': [str(e) for e in list(abnormal_edges.keys())],
                        'normal_edges': [str(e) for e in list(normal_edges.keys())],
                        'abnormal_edge_types': list(set([e.get('edge_type') for e in abnormal_entries if 'edge_type' in e])),
                        'abnormal_edge_stats': abnormal_edge_stats,
                        'normal_edge_stats': normal_edge_stats
                    }

            # Save anomalous IP behavior log for this window
            abnormal_log_path = os.path.join(netflow_log_dir, f"abnormal_ip_log_{time_interval}.json")
            with open(abnormal_log_path, "w") as f_ablog:
                json.dump(abnormal_ip_log, f_ablog, ensure_ascii=False, indent=2)
        else:
            ip_behavior_process_logger.info(
                f"Time window: {time_interval}, thr: {thr}, IP behavior log not found: {iplog_path}")

        # Incrementally construct the queues
        added_que_flag = False
        for hq in history_list:
            for his_tw in hq:
                if cal_set_rel(current_tw['nodeset'], his_tw['nodeset'], node_IDF, tw_list) != 0 and current_tw['name'] != his_tw['name']:
                    hq.append(copy.deepcopy(current_tw))
                    added_que_flag = True
                    break
                if added_que_flag:
                    break
        if added_que_flag is False:
            temp_hq = [copy.deepcopy(current_tw)]
            history_list.append(temp_hq)

        index_count += 1


        logger.info(f"Average loss: {loss_avg}")
        logger.info(f"Num of anomalous edges within the time window: {count}")
        logger.info(f"Percentage of anomalous edges: {count / len(edge_list) if len(edge_list) > 0 else 0}")
        logger.info(f"Anomalous node count: {len(node_set)}")
        logger.info(f"Anomalous edge count: {len(edge_set)}")
        logger.info("**************************************************")

    return history_list


if __name__ == "__main__":
    logger.info("Start logging.")

    node_IDF, tw_list = compute_IDF()

    # Validation date
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_5/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_5_history_list")

    # Testing date
    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_6/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_6_history_list")

    history_list = anomalous_queue_construction(
        node_IDF=node_IDF,
        tw_list=tw_list,
        graph_dir_path=f"{artifact_dir}/graph_4_7/"
    )
    torch.save(history_list, f"{artifact_dir}/graph_4_7_history_list")