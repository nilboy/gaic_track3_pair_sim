FROM registry.cn-shanghai.aliyuncs.com/gaic_3/gaic_3_xiaohua:base
COPY code /root/code
COPY user_data_submit /root/user_data
WORKDIR /root/code
CMD ["sh", "run.sh"]
