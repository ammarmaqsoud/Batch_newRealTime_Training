from time import sleep
from json import dumps
from kafka import KafkaProducer
import numpy as np
import math
import datetime as dt
import sys



if __name__ == '__main__':
    from kafka import KafkaAdminClient
    from kafka.admin import NewTopic

    # Configuration
    kafka_broker = "localhost:9092"
    topic_name = "speed_topic"

# #-----------------ReCreate thE topic------------------------
#     # Create an Admin client
#     admin_client = KafkaAdminClient(bootstrap_servers=kafka_broker)
#     # Delete the topic
#     try:
#         admin_client.delete_topics([topic_name])
#         print(f"Deleted topic: {topic_name}")
#     except Exception as e:
#         print(f"Error deleting topic: {e}")
#
#     # Recreate the topic
#     new_topic = NewTopic(name=topic_name, num_partitions=1, replication_factor=1)
#     try:
#         admin_client.create_topics([new_topic])
#         print(f"Recreated topic: {topic_name}")
#     except Exception as e:
#         print(f"Error recreating topic: {e}")
#     # -----------------ReCreate thE topic------------------------





    # if len(sys.argv) != 4:
    #     print("Usage: streamsourcedata.py <hostname:port> <topic>", file=sys.stderr)
    #     sys.exit(-1)
    broker = "localhost:9092"#sys.argv[1]
    topic = "speed_topic"#sys.argv[2]
    src_file = "../Data/DailyDelhiClimateTrain.csv" #str(sys.argv[3])
    f = open(src_file,"+r")
    lines = f.readlines()
    for line in lines:
        x = line.split(",")
        msg = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  + ',' + x[1]
        msg = msg.replace("\n","")
        # print()
        producer = KafkaProducer(bootstrap_servers=[broker],
                             value_serializer=lambda x:
                             dumps(x).encode('utf-8'))
        if ('meantemp' in msg):
            continue

        producer.send(topic, value=msg)
        print(msg)
        # sleep(0.5)



#
 #
 #
 #
 # cnt=0
 #    window=""
 #    for line in lines:
 #        x = line.split(",")
 #        if cnt==0:
 #            cnt=1
 #            continue
 #        msg = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  + ',' + x[1]
 #        msg = msg.replace("\n","")
 #        if window=="":
 #            window = x[1]
 #        else:
 #            window=window+","+x[1]
 #        # print()
 #        if cnt%7==0:
 #            producer = KafkaProducer(bootstrap_servers=[broker],
 #                             value_serializer=lambda x:
 #                             dumps(x).encode('utf-8'))
 #            producer.send(topic, value=msg)
 #            print(window)
 #            # sleep(1)
 #            cnt=1
 #        cnt=cnt+1
