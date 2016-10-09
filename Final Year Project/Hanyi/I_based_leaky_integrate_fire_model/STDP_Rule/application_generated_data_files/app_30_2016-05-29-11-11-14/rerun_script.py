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
    "192.168.240.253_appData_1_0_3.dat",
    1, 0, 3, 1612972032))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_1_7.dat",
    0, 1, 7, 1612972032))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_9.dat",
    0, 0, 9, 1612972032))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_1_4.dat",
    0, 1, 4, 1612972264))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_1_1.dat",
    0, 1, 1, 1612974608))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_6.dat",
    1, 1, 6, 1612972032))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_8.dat",
    0, 0, 8, 1612972780))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_1_8.dat",
    0, 1, 8, 1612974840))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_1.dat",
    1, 0, 1, 1612972780))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_1.dat",
    1, 1, 1, 1612974376))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_6.dat",
    1, 0, 6, 1612973528))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_7.dat",
    1, 1, 7, 1612974608))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_2.dat",
    1, 1, 2, 1612974840))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_15.dat",
    1, 1, 15, 1612977184))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_7.dat",
    1, 0, 7, 1612974276))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_12.dat",
    1, 1, 12, 1612977416))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_11.dat",
    1, 1, 11, 1612979760))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_7.dat",
    0, 0, 7, 1612973528))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_4.dat",
    1, 0, 4, 1612975024))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_6.dat",
    0, 0, 6, 1612974276))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_5.dat",
    1, 0, 5, 1612975772))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_1_5.dat",
    0, 1, 5, 1612977184))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_5.dat",
    0, 0, 5, 1612975024))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_1_2.dat",
    0, 1, 2, 1612977416))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_14.dat",
    1, 1, 14, 1612979992))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_4.dat",
    0, 0, 4, 1612975772))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_5.dat",
    1, 1, 5, 1612982336))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_3.dat",
    0, 0, 3, 1612976520))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_10.dat",
    1, 0, 10, 1612976520))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_16.dat",
    0, 0, 16, 1612977268))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_2.dat",
    0, 0, 2, 1612978016))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_11.dat",
    1, 0, 11, 1612977268))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_10.dat",
    1, 1, 10, 1612982568))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_1.dat",
    0, 0, 1, 1612978756))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_9.dat",
    1, 1, 9, 1612984912))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_8.dat",
    1, 0, 8, 1612978008))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_16.dat",
    1, 1, 16, 1612985144))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_4.dat",
    1, 1, 4, 1612987488))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_9.dat",
    1, 0, 9, 1612978756))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_1_6.dat",
    0, 1, 6, 1612979760))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_15.dat",
    0, 0, 15, 1612979504))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_1_3.dat",
    0, 1, 3, 1612982104))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_14.dat",
    1, 0, 14, 1612979504))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_14.dat",
    0, 0, 14, 1612980252))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_15.dat",
    1, 0, 15, 1612981848))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_13.dat",
    0, 0, 13, 1612981000))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_12.dat",
    1, 0, 12, 1612982080))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_12.dat",
    0, 0, 12, 1612981748))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_3.dat",
    1, 1, 3, 1612989832))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_13.dat",
    1, 0, 13, 1612982828))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_13.dat",
    1, 1, 13, 1612990064))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_11.dat",
    0, 0, 11, 1612982496))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_1_8.dat",
    1, 1, 8, 1612990296))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_16.dat",
    1, 0, 16, 1612983060))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_1_0_2.dat",
    1, 0, 2, 1612985404))
application_data.append(ReloadApplicationData(
    "192.168.240.253_appData_0_0_10.dat",
    0, 0, 10, 1612983244))
reload_routing_table = ReloadRoutingTable()
routing_tables.add_routing_table(reload_routing_table.reload("picked_routing_table_for_1_1"))
reload_routing_table = ReloadRoutingTable()
routing_tables.add_routing_table(reload_routing_table.reload("picked_routing_table_for_0_1"))
reload_routing_table = ReloadRoutingTable()
routing_tables.add_routing_table(reload_routing_table.reload("picked_routing_table_for_1_0"))
reload_routing_table = ReloadRoutingTable()
routing_tables.add_routing_table(reload_routing_table.reload("picked_routing_table_for_0_0"))
binaries.add_subsets("C:\\Python27\\lib\\site-packages\\spynnaker\\pyNN\\model_binaries\\IF_curr_exp.aplx", CoreSubsets([CoreSubset(0, 1, [1, 3, 5, 7, ]),CoreSubset(1, 1, [1, 3, 5, 7, 9, 11, 13, 15, ]),CoreSubset(1, 0, [13, 15, ]),]))
binaries.add_subsets("C:\\Python27\\lib\\site-packages\\spynnaker\\pyNN\\model_binaries\\IF_curr_exp_stdp_mad_nearest_pair_additive.aplx", CoreSubsets([CoreSubset(0, 1, [8, 2, 4, 6, ]),CoreSubset(1, 1, [2, 4, 6, 8, 10, 12, 14, 16, ]),CoreSubset(1, 0, [16, 14, ]),]))
binaries.add_subsets("C:\\Python27\\lib\\site-packages\\spinn_front_end_common\\common_model_binaries\\reverse_iptag_multicast_source.aplx", CoreSubsets([CoreSubset(1, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ]),CoreSubset(0, 0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, ]),]))
vertex = ReloadBufferedVertex("Population 2:0:0", [(2, "Population 2_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 1))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 3:0:0", [(2, "Population 3_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 2))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 6:0:0", [(2, "Population 6_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 3))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 7:0:0", [(2, "Population 7_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 4))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 10:0:0", [(2, "Population 10_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 5))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 11:0:0", [(2, "Population 11_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 6))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 14:0:0", [(2, "Population 14_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 7))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 15:0:0", [(2, "Population 15_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 8))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 18:0:0", [(2, "Population 18_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 9))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 19:0:0", [(2, "Population 19_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 10))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 22:0:0", [(2, "Population 22_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 11))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 23:0:0", [(2, "Population 23_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 12))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 26:0:0", [(2, "Population 26_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 13))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 27:0:0", [(2, "Population 27_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 14))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 30:0:0", [(2, "Population 30_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 15))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 31:0:0", [(2, "Population 31_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 0, 0, 16))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 34:0:0", [(2, "Population 34_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 1))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 35:0:0", [(2, "Population 35_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 2))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 38:0:0", [(2, "Population 38_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 3))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 39:0:0", [(2, "Population 39_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 4))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 42:0:0", [(2, "Population 42_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 5))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 43:0:0", [(2, "Population 43_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 6))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 46:0:0", [(2, "Population 46_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 7))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 47:0:0", [(2, "Population 47_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 8))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 50:0:0", [(2, "Population 50_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 9))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 51:0:0", [(2, "Population 51_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 10))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 54:0:0", [(2, "Population 54_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 11))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 
vertex = ReloadBufferedVertex("Population 55:0:0", [(2, "Population 55_0_0_2", 1048576) ])
buffered_placements.add_placement(Placement(vertex, 1, 0, 12))
buffered_tags.add_ip_tag(IPTag("192.168.240.253", 0, "0.0.0.0", 17896, True), vertex) 

reloader = Reload(machine_name, machine_version, reports_states, bmp_details, down_chips, down_cores, number_of_boards, height, width, auto_detect_bmp,enable_reinjection)
if len(socket_addresses) > 0:
    reloader.execute_notification_protocol_read_messages(socket_addresses, None, os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_output_database.db"))
reloader.reload_application_data(application_data)
reloader.reload_routes(routing_tables)
reloader.reload_tags(iptags, reverse_iptags)
reloader.reload_binaries(binaries)
reloader.enable_buffer_manager(buffered_placements, buffered_tags)
reloader.restart(binaries, 60100, 1, turn_off_machine=True)
