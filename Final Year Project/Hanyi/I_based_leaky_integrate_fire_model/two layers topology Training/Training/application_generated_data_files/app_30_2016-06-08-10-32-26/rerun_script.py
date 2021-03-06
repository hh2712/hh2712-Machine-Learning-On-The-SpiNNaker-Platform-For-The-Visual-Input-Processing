"""
general reload script - note that imports are required so don't remove them!
"""

# pacman imports
from pacman.model.placements.placements import Placements
from pacman.model.placements.placement import Placement
from pacman.model.routing_info.routing_info import RoutingInfo
from pacman.model.routing_info.subedge_routing_info import SubedgeRoutingInfo
from pacman.model.routing_tables.multicast_routing_tables import \
    MulticastRoutingTables
from pacman.model.tags.tags import Tags

# spinnman imports
from spinnman.model.core_subsets import CoreSubsets
from spinnman.model.core_subset import CoreSubset

# spinnmachine imports
from spinn_machine.tags.iptag import IPTag
from spinn_machine.tags.reverse_iptag import ReverseIPTag

# front end common imports
from spinn_front_end_common.utilities.report_states import ReportState
from spinn_front_end_common.utilities.reload.reload import Reload
from spinn_front_end_common.utilities.reload.reload_application_data \
    import ReloadApplicationData
from spinn_front_end_common.utilities.executable_targets \
    import ExecutableTargets
from spinn_front_end_common.utilities.reload.reload_routing_table import \
    ReloadRoutingTable
from spinn_front_end_common.utilities.reload.reload_buffered_vertex import \
    ReloadBufferedVertex
from spinn_front_end_common.utilities.notification_protocol.\
    socket_address import SocketAddress

# general imports
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)-15s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"))

application_data = list()
binaries = ExecutableTargets()
iptags = list()
reverse_iptags = list()
buffered_tags = Tags()
buffered_placements = Placements()

routing_tables = MulticastRoutingTables()
# database params
socket_addresses = list()

reports_states = ReportState(False, False, False, False, False,
                             False, False, False, False, False)
machine_name = "192.168.240.253"
machine_version = 3
bmp_details = "None"
down_chips = "None"
down_cores = "None"
number_of_boards = 1
height = None
width = None
auto_detect_bmp = False
enable_reinjection = True
iptags.append(
    IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True)) 
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_1.dat",
    0, 0, 1, 1612972032))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_2.dat",
    0, 0, 2, 1612998780))
reload_routing_table = ReloadRoutingTable()
routing_tables.add_routing_table(reload_routing_table.reload("picked_routing_table_for_0_0"))
binaries.add_subsets("C:\\Python27\\lib\\site-packages\\spynnaker\\pyNN\\model_binaries\\IF_curr_exp.aplx", CoreSubsets([CoreSubset(0, 0, [2, ]),]))
binaries.add_subsets("C:\\Python27\\lib\\site-packages\\spinn_front_end_common\\common_model_binaries\\reverse_iptag_multicast_source.aplx", CoreSubsets([CoreSubset(0, 0, [1, ]),]))
vertex = ReloadBufferedVertex("inputSpikes_On:0:255", [(2, "inputSpikes_On_0_255_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 1))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 

reloader = Reload(machine_name, machine_version, reports_states, bmp_details, down_chips, down_cores, number_of_boards, height, width, auto_detect_bmp,enable_reinjection)
if len(socket_addresses) > 0:
    reloader.execute_notification_protocol_read_messages(socket_addresses, None, os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_output_database.db"))
reloader.reload_application_data(application_data)
reloader.reload_routes(routing_tables)
reloader.reload_tags(iptags, reverse_iptags)
reloader.reload_binaries(binaries)
reloader.enable_buffer_manager(buffered_placements, buffered_tags)
reloader.restart(binaries, 8800, 1, turn_off_machine=True)
