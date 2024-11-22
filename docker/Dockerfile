FROM ghcr.io/nvidia/jax:nightly-2023-09-09

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Install python3, pip, and vim
RUN apt-get -y update
RUN apt-get -y install python3 python3-pip git
RUN apt-get -y install vim

# Install basic libraries
RUN pip3 install --upgrade pip
RUN pip3 install numpy matplotlib scipy

# Install Jax
# RUN pip3 install --upgrade "jax[cuda11]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install Jax-related libraries
RUN pip3 install optax

# Install miscellaneous libraries
RUN pip3 install tqdm
RUN apt-get -y install python3-tk

# Install Mujoco and add to path
RUN apt-get install -y wget
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN mkdir -p ~/.mujoco/mujoco210
RUN tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
RUN rm mujoco210-linux-x86_64.tar.gz

# Install mujoco-py dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y libxrender1 libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev xpra patchelf libglfw3-dev libglfw3

# Install mujoco-py and OpenAI Gym
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
RUN pip3 install -U Cython==3.0.0a10
ADD mujoco-py /root/mujoco-py
RUN cd /root/mujoco-py && pip3 install --no-cache-dir -r requirements.txt && pip3 install --no-cache-dir -r requirements.dev.txt && python3 setup.py build && python3 setup.py install
RUN rm -rf /root/mujoco-py
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
RUN pip3 install gym
RUN python3 -c "import mujoco_py; exit"   # Import for preliminary compile
RUN apt-get install -y ffmpeg && pip3 install moviepy

# Install dill
RUN pip3 install dill

# Environment setup
RUN echo '' >> /root/.bashrc
RUN echo 'alias python=python3' >> /root/.bashrc

CMD /bin/bash
