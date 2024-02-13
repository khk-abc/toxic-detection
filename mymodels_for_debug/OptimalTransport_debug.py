import torch
import torch.nn as nn
import ot
from multiprocessing import Pool
import torch.multiprocessing as mp
from .sinkhorn_log_debug import sinkhorn_log,sinkhorn_log_batch,sinkhorn_stabilized_batch
from time import time
import torch.nn.functional as F

# mp.set_start_method('spawn', force=True)

def timer(func):
    def func_wrapper(*args, **kwargs):
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper

class OptimalTransport(nn.Module):
    def __init__(self,train_cost_matrix=False,train_cost_matrix_v2=False):
        super(OptimalTransport, self).__init__()
        self.mask_matrix = None
        self.epsilon = 1e-2  # TODO epsilon--> trainable parameter

        #TODO 将任务特征加入进来，与source特征、target特征输入模块中，进行可训练的条件代价
        self.train_cost_matrix=train_cost_matrix
        self.train_cost_matrix_v2=train_cost_matrix_v2

        if self.train_cost_matrix:
            self.train_conditional_cost = nn.Sequential(
                nn.Linear(768,1),
                nn.Sigmoid(), #TODO 是否不去除sigmoid，不强制将其映射到0-1之间
            )
            self.task_fea = nn.Embedding(4,768)

            self.alpha = nn.Parameter(torch.tensor(0.05),requires_grad=True)


        elif self.train_cost_matrix_v2:
            self.train_conditional_cost_v2 = nn.Sequential(
                nn.Linear(768,1),
            )
            self.task_fea_v2 = nn.Embedding(4,768)
            self.alpha_v2 = nn.Parameter(torch.tensor(0.05),requires_grad=True)

    def compute_kernel_for_matrix_batch(self,matrix, h=1):
        '''
        compute Gaussian kernel matrices
        Parameters
        ----------
        matrix: pairwise distance matrix
        h : bandwidth
        Returns
        ----------
        K: source-target kernel
        '''
        std = torch.std(matrix,dim=[-1,-2],keepdim=True)
        # std = torch.sqrt((matrix ** 2).mean(dim=[-1,-2],keepdim=True) / 2)
        h = h * std
        # Gaussian kernel (without normalization)
        K = torch.exp(-(matrix / h) ** 2 / 2)
        return K

    def _comput_cost_matrix_batch(self, source_features, target_features, metric='cosine',task_fea=None,task_id=None,use_kernel_cost=False):
        bz = source_features.shape[0]
        if metric=='cosine':
            cost_matrix = (1 - torch.cos(torch.einsum('bsd,btd->bst', source_features, target_features))) / 2
        elif metric == 'l1':
            cost_matrix = (torch.abs(source_features.unsqueeze(2).repeat(1,1,target_features.shape[1],1) -
                                     target_features.unsqueeze(1).repeat(1,source_features.shape[1],1,1))).sum(dim=-1)
            cost_matrix = cost_matrix/cost_matrix.sum(dim=-1,keepdim=True)
        elif metric == 'l2':
            cost_matrix = torch.sqrt(((source_features.unsqueeze(2).repeat(1,1,target_features.shape[1],1) -
                                       target_features.unsqueeze(1).repeat(1,source_features.shape[1],1,1)) ** 2).sum(dim=-1))
            cost_matrix = cost_matrix / cost_matrix.sum(dim=-1, keepdim=True)
        elif metric=='kl':
            cost_matrix = (F.kl_div(input=torch.sigmoid(source_features.unsqueeze(2).repeat(1,1,target_features.shape[1],1)),
                                    target=torch.sigmoid(target_features.unsqueeze(1).repeat(1,source_features.shape[1],1,1)),
                                    reduction='none').mean(dim=-1)
                           - F.kl_div(input=torch.sigmoid(source_features),
                                      target=torch.sigmoid(source_features),
                                      reduction='none').mean(dim=-1,keepdim=True).repeat(1,1,target_features.shape[1]))
        elif metric=='kl_l2':
            cost_matrix = torch.sqrt(
                ((source_features.unsqueeze(2).repeat(1,1,target_features.shape[1],1) -
                  target_features.unsqueeze(1).repeat(1,source_features.shape[1],1,1)) ** 2).sum(dim=-1))
            cost_matrix = cost_matrix / cost_matrix.sum(dim=-1, keepdim=True)

            # print("l2",cost_matrix[0])

            differ =F.kl_div(input=torch.sigmoid(source_features.unsqueeze(2).repeat(1,1,target_features.shape[1],1)),
                              target=torch.sigmoid(target_features.unsqueeze(1).repeat(1,source_features.shape[1],1,1)),
                              reduction='none').mean(dim=-1)
            base = F.kl_div(input=torch.sigmoid(source_features),
                            target=torch.sigmoid(source_features),
                            reduction='none').mean(dim=-1,keepdim=True).repeat(1,1,target_features.shape[1])
            # print("kl: ",(differ - base)[0])
            cost_matrix = cost_matrix.add(differ - base)

        if use_kernel_cost:
            cost_matrix = self.compute_kernel_for_matrix_batch(matrix=cost_matrix,h=1)

        if self.train_cost_matrix:
            assert task_fea is not None or task_id is not None
            task_fea = self.task_fea(torch.tensor(task_id).repeat(bz).to(source_features.device))

            # TODO: 是否考虑将计算的基于相似度的代价矩阵也作为可训练代价矩阵的部分输入？
            contional_combined_fea = source_features.unsqueeze(2).repeat(1,1,target_features.shape[1],1)+\
                                     target_features.unsqueeze(1).repeat(1,source_features.shape[1],1,1)+\
                                     task_fea.unsqueeze(1).unsqueeze(1).repeat(1,source_features.shape[1],
                                                                               target_features.shape[1],1)

            cost_matrix = self.alpha*cost_matrix.add(self.train_conditional_cost(contional_combined_fea).squeeze(dim=-1))

        elif self.train_cost_matrix_v2:
            assert task_fea is not None or task_id is not None
            task_fea = self.task_fea_v2(torch.tensor(task_id).repeat(bz).to(source_features.device))

            # TODO: 是否考虑将计算的基于相似度的代价矩阵也作为可训练代价矩阵的部分输入？
            contional_combined_fea = source_features.unsqueeze(2).repeat(1,1,target_features.shape[1],1)+\
                                     target_features.unsqueeze(1).repeat(1,source_features.shape[1],1,1)+\
                                     task_fea.unsqueeze(1).unsqueeze(1).repeat(1,source_features.shape[1],
                                                                               target_features.shape[1],1)

            # contional_combined_fea = contional_combined_fea + cost_matrix.unsqueeze(-1)

            cost_matrix = self.alpha_v2*cost_matrix.add(self.train_conditional_cost_v2(contional_combined_fea).squeeze(dim=-1))


        return cost_matrix


    def pairwise_dis_batch(self,x, y, metric='cosine'):
        if metric == 'l2':
            pairwise_dis = torch.sqrt(((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(dim=-1))
            pairwise_dis = pairwise_dis/pairwise_dis.sum(dim=-1,keepdim=True)
        elif metric == 'l1':
            pairwise_dis = (torch.abs(x[:, :, None, :] - y[:, None, :, :])).sum(dim=-1)
            pairwise_dis = pairwise_dis / pairwise_dis.sum(dim=-1, keepdim=True)
        elif metric == 'cosine':
            pairwise_dis = (1 - torch.cos(torch.einsum('bsd,btd->bst', x, y))) / 2
        return pairwise_dis


    def compute_kernel_batch(self,Cx, Cy, h):
        '''
        compute Gaussian kernel matrices
        Parameters
        ----------
        Cx: source pairwise distance matrix
        Cy: target pairwise distance matrix
        h : bandwidth
        Returns
        ----------
        Kx: source kernel
        Ky: targer kernel
        '''
        # print(Cx.shape)
        bz,source_len,source_len = Cx.shape
        bz,target_len,target_len = Cy.shape
        # [bz,]
        std1 = torch.sqrt((Cx ** 2).sum(dim=-1).sum(dim=-1) / (2*source_len*source_len))
        std2 = torch.sqrt((Cy ** 2).sum(dim=-1).sum(dim=-1) / (2*target_len*target_len))
        h1 = h * std1
        h2 = h * std2
        # print(h1.shape)
        # Gaussian kernel (without normalization)
        Kx = torch.exp(-(Cx / h1.unsqueeze(dim=-1).unsqueeze(dim=-1)) ** 2 / 2)
        Ky = torch.exp(-(Cy / h2.unsqueeze(dim=-1).unsqueeze(dim=-1)) ** 2 / 2)
        return Kx, Ky

    def marginal_prob_unform(self, bz, source_length, target_length):
        source_u = torch.ones(bz, source_length) / source_length
        target_u = torch.ones(bz, target_length) / target_length
        return source_u, target_u


    def ot_log_sinkhorn(self, source, target, max_iter, epsilon=0.1, thresh=0.01, cost_matrix=None, speedup=False,
                        detach=False,lam=1,metric='cosine',task_fea=None, task_id=None,use_kernel_cost=False,
                     Adj_matrix=None, mask=None, reduction=None, position_filter=None):
        # TODO epsilon--> trainable parameter
        if detach:
            source = source.detach()
            target = target.detach()


        if cost_matrix is None:
            # [bz*dims,source_len,target_len]
            cost_matrix = self._comput_cost_matrix_batch(source_features=source.unsqueeze(-1),
                                                         target_features=target.unsqueeze(-1),
                                                         task_fea=task_fea,
                                                         task_id=task_id,
                                                         metric=metric,
                                                         use_kernel_cost=use_kernel_cost,
                                                         )
            if metric!='cosine':
                cost_matrix = cost_matrix/cost_matrix.max(dim=-1,keepdim=True)[0]
            # print(cost_matrix)

        bz  = source.shape[0]
        # print(source.shape)
        # print(source)
        # print(cost_matrix)

        if speedup:
            # TODO : 2024.01.26: 使用ot计算时不需要在计算初始分布
            # [bz*dims,source_len,target_len]
            trasport_matrix = sinkhorn_log_batch(source, target, cost_matrix,
                                                 use_kernel=False,
                                                 lam=lam,
                                                 reg=epsilon, numItermax=max_iter,
                                                 stopThr=thresh,
                                                 period=10,
                                                 )

        else:
            trasport_matrix = []
            for b in range(bz):
                # TODO : 2024.01.26: 使用ot计算时不需要在计算初始分布
                trasport_matrix.append(ot.sinkhorn(source[b],target[b],cost_matrix[b],
                                                   reg=epsilon,numItermax=max_iter,stopThr=thresh,
                                                   method='sinkhorn_log'))


            trasport_matrix = torch.stack(trasport_matrix,dim=0)

        assert trasport_matrix.shape == cost_matrix.shape
        cost_distance = torch.mul(cost_matrix, trasport_matrix).mean()

        return cost_distance,trasport_matrix,None,None,None


    def ot_log_sinkhorn_with_kernel(self, source, target, max_iter, epsilon=0.1, thresh=0.01,
                                    h=1,lam = 1, max_MI_iter=50, cost_matrix=None, speedup=False,
                                    detach=False,metric='cosine', task_fea=None,task_id=None,use_kernel_cost=False,
                     Adj_matrix=None, mask=None, reduction=None, position_filter=None):
        # TODO epsilon--> trainable parameter
        if detach:
            source = source.detach()
            target = target.detach()

        bz = source.shape[0]
        source_length, target_length = source.shape[1], target.shape[1]
        source_u, target_u = self.marginal_prob_unform(bz=bz,
                                                       source_length=source_length,
                                                       target_length=target_length)
        source_u,target_u = source_u.to(source.device),target_u.to(source.device)

        if cost_matrix is None:
            cost_matrix = self._comput_cost_matrix_batch(source_features=source,
                                                         target_features=target,
                                                         task_fea=task_fea,
                                                         task_id=task_id,
                                                         metric=metric,
                                                         use_kernel_cost=use_kernel_cost,
                                                         )

        # pairwise_dis_source = self.pairwise_dis_batch(source,source,metric=metric)
        # pairwise_dis_target = self.pairwise_dis_batch(target,target,metric=metric)
        # kernel_source,kernel_target = self.compute_kernel_batch(pairwise_dis_source,pairwise_dis_target,h=h)
        kernel_source, kernel_target = None, None

        if speedup:
            trasport_matrix = sinkhorn_log_batch(source_u, target_u, cost_matrix,
                                                 use_kernel=True,
                                                 kernel_source=kernel_source, kernel_target=kernel_target, lam=lam,
                                                 step=1,
                                                 reg=epsilon, numItermax=max_iter, stopThr=thresh,
                                                 )
        else:
            trasport_matrix = []
            for b in range(bz):
                trasport_matrix.append(sinkhorn_log(source_u[b], target_u[b], cost_matrix[b],
                                                    use_kernel=True,
                                                    step=1,
                                                    kernel_source=kernel_source[b] if kernel_source is not None else None,
                                                    kernel_target=kernel_target[b] if kernel_target is not None else None,
                                                    lam=lam,
                                                    reg=epsilon, numItermax=max_iter, stopThr=thresh,
                                                    ))
            trasport_matrix = torch.stack(trasport_matrix,dim=0)

        assert trasport_matrix.shape==cost_matrix.shape
        cost_distance = torch.mul(cost_matrix,trasport_matrix).mean()

        return cost_distance,trasport_matrix,None,None,None
    # @timer
    def forward(self, source_features, target_features, task_fea=None, task_id=None, epsilon=0.1, thresh=1e-3, max_iter=100,use_OT_kernel=False,
                speedup=False,detach=False,metric='cosine',use_kernel_cost=False,
                cost_matrix=None, Adj_matrix=None, mask=None):
        """
        将起点特征source features传输到终点特征target features
        :param source_features: 最优传输的起点特征， 【bz, source_lens, dims】
        :param target_features: 最优传输的终点特征， 【bz, target_lens, dims】
        :param cost_matrix: 代价矩阵
        :return: transported target features 经过最优传输后的终点特征
        """

        if use_OT_kernel:
            cost_distance, trasport_matrix, cost_matrix, U, V = self.ot_log_sinkhorn_with_kernel(source=source_features,
                                                                                                 target=target_features,
                                                                                                 task_fea=task_fea,
                                                                                                 task_id=task_id,
                                                                                                 max_iter=max_iter,
                                                                                                 epsilon=epsilon,
                                                                                                 thresh=thresh,
                                                                                                 lam=1,
                                                                                                 cost_matrix=cost_matrix,
                                                                                                 Adj_matrix=Adj_matrix,
                                                                                                 mask=mask,
                                                                                                 speedup=speedup,
                                                                                                 detach=detach,
                                                                                                 metric=metric,
                                                                                                 use_kernel_cost=use_kernel_cost,
                                                                                                 )
        else:
            cost_distance, trasport_matrix, cost_matrix, U, V = self.ot_log_sinkhorn(source=source_features,
                                                                                     target=target_features,
                                                                                     task_fea=task_fea,
                                                                                     task_id=task_id,
                                                                                     max_iter=max_iter,
                                                                                     epsilon=epsilon,
                                                                                     thresh=thresh,
                                                                                     lam=1,
                                                                                     cost_matrix=cost_matrix,
                                                                                     Adj_matrix=Adj_matrix,
                                                                                     mask=mask,
                                                                                     speedup=speedup,
                                                                                     detach=detach,
                                                                                     metric=metric,
                                                                                     use_kernel_cost=use_kernel_cost
                                                                                     )

        return cost_distance, trasport_matrix, cost_matrix, U, V


if __name__ == '__main__':
    from matplotlib import pyplot as plt


    x1=torch.range(-5,5,0.1)
    u1=0
    sigma1=1
    y1 = torch.multiply(torch.pow(torch.sqrt(torch.tensor(2*3.1415))*sigma1,-1),torch.exp(-torch.pow(x1-u1,2)/2*sigma1))


    x2=torch.range(0,10,0.1)
    u2=5
    sigma2=0.5
    y2 = torch.multiply(torch.pow(torch.sqrt(torch.tensor(2*3.1415))*sigma2,-1),torch.exp(-torch.pow(x2-u2,2)/2*sigma2))

    model = OptimalTransport()
    cost_distance, trasport_matrix, cost_matrix, U, V = model(y1.unsqueeze(0).unsqueeze(-1), y2.unsqueeze(0).unsqueeze(-1),
                                                              epsilon=1e-1,thresh=1e-9)

    plt.figure()
    plt.plot(x1,y1,linewidth=2)
    plt.show()

    plt.figure()
    plt.plot(x2, y2, linewidth=2)
    plt.show()

    # plt.figure()
    # plt.bar(list(range(y.shape[1])),y[0,:,0])
    # plt.show()

    print(trasport_matrix)


