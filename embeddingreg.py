from utilities import AbstractNetwork, GeneralDatasetLoader
import torch
import warnings
from copy import deepcopy
import random
import torch.nn.functional as F 

class embedding(object):
    def __init__(self, model: AbstractNetwork, dataset: GeneralDatasetLoader, config, gpu):

        self.model = model
        self.dataset = dataset
        self.gpu = gpu

        self.is_conv = config.IS_CONVOLUTIONAL
        self.device = config.DEVICE
        self.batch_size = config.BATCH_SIZE
        self.first_batch = True
        self.incremental = config.IS_INCREMENTAL
        self.handle = None

        self.sample_size = config.CL_PAR.get('sample_size', 25)
        self.memorized_task_size = config.CL_PAR.get('memorized_task_size', 300)
        self.importance = config.CL_PAR.get('penalty_importance', 1)
        self.c = config.CL_PAR.get('c', 1)
        self.distance = config.CL_PAR.get('distance', 'euclidean')
        self.supervised = config.CL_PAR.get('supervised', False)
        self.normalize = config.CL_PAR.get('normalize', True)
        self.mul = config.CL_PAR.get('normalize', 1)

        self.online = config.CL_PAR.get('online', False)
        self.memory_size = config.CL_PAR.get('memory_size', -1)

        if 0 < self.memory_size < self.memorized_task_size:
            self.memorized_task_size = self.memory_size

        # Can be distance, usage, image_similarity or none
        self.weights_type = config.CL_PAR.get('weights_type', None)

        if self.weights_type == 'image_similarity':
            img_size = dataset[0][0].size()
            if self.is_conv:
                pass
            else:
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(28*28, 300),
                    torch.nn.ReLU(),
                    torch.nn.Linear(300, 200),
                    torch.nn.ReLU(),
                    torch.nn.Linear(200, 100),
                ).to(config.DEVICE)

        # Can be None, a tuple (a, b) or an integer. IF a tuple (a, b) is given then  these are the input and
        # output size of the layer, otherwise if a single integer is given then
        # it will be the dimension for both input and output
        external_embedding = config.CL_PAR.get('external_embedding', None)

        if external_embedding is not None:
            a = b = None
            if isinstance(external_embedding, int):
                a = external_embedding
                b = a
            elif isinstance(external_embedding, (tuple, list)):
                a, b = external_embedding

            if a is not None:
                self.external_embedding_layer = torch.nn.Linear(a, b, bias=False).to(self.device)

                torch.nn.init.orthogonal_(self.external_embedding_layer.weight.data)

                for param in self.external_embedding_layer.parameters():
                    param.requires_grad = False

                self.model.external_embedding_layer = self.external_embedding_layer

            else:
                warnings.warn("Possible values for external_embedding are integer or (integer, integer")

        else:
            self.external_embedding_layer = None

        # def hook(module, input, output):
        #     setattr(module, "_value_hook", output)
        #
        # for n, m in self.model.named_modules():
        #     if n != 'proj':
        #         m.register_forward_hook(hook)

        self._current_task = 0
        self.sample_size_task = 0
        self.batch_count = 0

        self.tasks = set()

        self.embeddings = {}
        self.embeddings_images = {}
        self.w = {}
        # self.embeddings_images = []

    def __call__(self, *args, **kwargs):

        if self.importance == 0:
            return self, 0

        current_task = kwargs['current_task']

        penalty = 0

        if current_task > 0:
            current_batch = kwargs['batch']

            self.batch_count += 1
            self.embedding_save(current_task)

            if not self.first_batch:# and self.batch_count % self.c == 0:
                penalty = self.embedding_drive(current_batch)
            else:
                self.first_batch = False

        return self, penalty

    def embedding_save(self, current_task):

        if current_task-1 not in self.tasks:
            self.importance *= self.mul
            self.first_batch = True
            self.tasks.add(current_task-1)

            self.batch_count = 0

            self.model.eval()
            # self.dataset.train_phase()

            # it = self.dataset.getIterator(self.memorized_task_size, task=current_task-1)

            # if self.incremental:
            #     if self.handle is not None:
            #         self.handle.remove()
            #
            #     mask = np.ones(self.model.output_size)
            #     for t in self.tasks:
            #         for i in self.dataset.task_mask(t):
            #             mask[i] = 0
            #     self.handle = self.model.classification_layer.register_backward_hook(self.layer_freeze(torch.from_numpy(mask).float().to(self.device)))
            #     # self.w.extend([1] * self.memorized_task_size)

            # images, _ = next(it)

            # input = images.to(self.device)
            embs = None

            with torch.no_grad():

                for step, ((y1, y2), _) in enumerate(self.dataset):
                    y1 = y1.cuda(self.gpu, non_blocking=True)
                    y2 = y2.cuda(self.gpu, non_blocking=True)
                    o_y1, o_y2 = self.model.module.embedding(y1, y2)
                    # compute average for embeddings
                    mid_out = [o_y1, o_y2]
                    output = torch.stack(mid_out).mean(dim=0)
                    # compute average for images
                    img_mid_output = [y1, y2]
                    images = torch.stack(img_mid_output).mean(dim=0)

                    if self.external_embedding_layer is not None:
                        output = self.external_embedding_layer(output)

                    if self.normalize:
                        output = F.normalize(output, p=2, dim=1)

                    embeddings = output.cpu()

                    if embs is None:
                        embs = embeddings
                    else:
                        embs = torch.cat((embs, embeddings), 0)

                if self.supervised:
                    self.embeddings[current_task-1] = embs.cpu()
                    self.embeddings_images[current_task-1] = images.cpu()

                    c = 1
                    # if self.weights_type is not None:
                    #     if self.weights_type == 'distance':
                    #         c = 0.1

                    w = [c] * self.memorized_task_size

                    for t in self.w.keys():
                        self.w[t] = deepcopy(w)

                    self.w[current_task-1] = deepcopy(w)

                else:
                    if len(self.embeddings) == 0:
                        self.embeddings[0] = embs.cpu()
                        self.embeddings_images[0] = images.cpu()
    
                    else:
                        self.embeddings[0] = torch.cat((self.embeddings[0], embs.cpu()), 0)
                        self.embeddings_images[0] = torch.cat((self.embeddings_images[0], images.cpu()), 0)

                    c = 1
                    # if self.weights_type is not None:
                    #     if self.weights_type == 'distance':
                    #         c = 0.1

                    # self.w[0] = [c] * self.embeddings[0].size()[0]
                    self.w[0] = [c] * 32


                # if self.embeddings is None or self.online:
                #     self.embeddings = embeddings
                #     self.embeddings_images = images
                #     self.w = [1] * self.memorized_task_size
                # else:
                #     self.embeddings = torch.cat((self.embeddings, embeddings), 0)
                #     self.embeddings_images = torch.cat((self.embeddings_images, images), 0)
                #     self.w = [1] * self.embeddings.size()[0]
                #
                # if 0 < self.memory_size < len(self.w):
                #     self.embeddings = self.embeddings[-self.memory_size:]
                #     self.embeddings_images = self.embeddings_images[-self.memory_size:]
                #     self.w = self.w[-self.memory_size:]

            self.model.train()
            # self.dataset.train_phase()

    def embedding_drive(self, current_batch):
        self.model.eval()

        for t in self.embeddings:

            to_back = None

            w = self.w[t]
            # print("w: ", w)

            idx = range(len(w))

            # w = self.w
            if self.weights_type == 'usage':
                # ws = np.sum(self.w)
                w = [1/wc for wc in w]

            if self.supervised:
                ss = self.sample_size
            else:
                ss = self.sample_size * len(self.tasks)

            idx = random.choices(idx, k=ss, weights=w)

            for i in idx:

                # print("emb: ", self.embeddings[0].shape, self.embeddings[0])
                # print()
                # print()
                # print("emb_imgs: ", self.embeddings_images[0].shape, self.embeddings_images[0])
                img = self.embeddings_images[t][i].unsqueeze(0).to(self.device)
                embeddings = self.embeddings[t][i].unsqueeze(0)

                new_embeddings1, new_embeddings2 = self.model.module.embedding(img, img)
                new_embeddings = torch.stack([new_embeddings1, new_embeddings2]).mean(dim=0)
                if self.external_embedding_layer is not None:
                    new_embeddings = self.external_embedding_layer(new_embeddings)

                if self.normalize:
                    new_embeddings = F.normalize(new_embeddings, p=2, dim=1)

                new_embeddings = new_embeddings.cpu()

                if self.distance == 'euclidean':
                    dist = (embeddings - new_embeddings).norm(p=None, dim=1)
                elif self.distance == 'cosine':
                    cosine = torch.nn.functional.cosine_similarity(embeddings, new_embeddings)
                    dist = 1-cosine

                if to_back is None:
                    to_back = dist
                else:
                    to_back = torch.cat((to_back, dist), 0)

                # dist = dist.mean() * self.importance
                # dist.backward()

                if self.weights_type is not None:
                    if self.weights_type == 'distance':
                        dist = dist.detach().cpu().numpy()
                        # for j, i in enumerate(idx):
                        self.w[t][i] = float(dist)

                    elif self.weights_type == 'usage':
                        # for j, i in enumerate(idx):
                        self.w[t][i] += 1

                    elif self.weights_type == 'image_similarity':
                        # compute images average again
                        current_batch_avg = torch.stack([current_batch[0], current_batch[1]]).mean(dim=0)
                        current_images = self.encoder(current_batch_avg)
                        old_images = self.encoder(img)

                        with torch.no_grad():
                            current_images = current_images / current_images.norm(dim=1)[:, None]
                            old_images = old_images / old_images.norm(dim=1)[:, None]
                            dist = torch.mm(current_images, old_images.transpose(0, 1))
                            dist = (1 - dist.mean(dim=1)).cpu().numpy()

                        # for j, i in enumerate(idx):
                        self.w[t][i] = dist

            to_back = to_back.mean() * self.importance
            # print(to_back, torch.log(to_back))
            to_back.backward()
            # l = torch.log(-to_back)
            # l.backward()

        self.model.train()
        return to_back.item()