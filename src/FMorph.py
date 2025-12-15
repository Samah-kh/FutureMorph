import torch
from .utils import load_model_and_plot_losses,plot_slices
from .utils import wrap_seg_trilinear, WrapSeg
from .loss import MSE,MultiClassDiceLoss
import torch.nn as nn
import os
import torchvision



def log_example_images(writer, moving, fixed, pred, step, slice_idx=None):
    """
    moving, fixed, pred : torch.Tensor [B,1,H,W,D]
    slice_idx : int, pick a slice along axial (default = middle slice)
    """
    moving, fixed, pred = moving.detach().cpu(), fixed.detach().cpu(), pred.detach().cpu()
    if slice_idx is None:
        slice_idx = moving.shape[-1] // 2

    # take middle slice in axial plane
    mv = moving[0,0,:,:,slice_idx]
    fx = fixed[0,0,:,:,slice_idx]
    pr = pred[0,0,:,:,slice_idx]

    # stack into a grid
    grid = torchvision.utils.make_grid([mv.unsqueeze(0), fx.unsqueeze(0), pr.unsqueeze(0)],
                                       nrow=3, normalize=True, scale_each=True)
    writer.add_image("Examples/Mv-Fx-Pred", grid, step)

def save_checkpoint(model, optimizer, epoch, losses_dict, save_path="checkpoints/latest.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses_dict
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint at epoch {epoch} to {save_path}")



class FMorph(nn.Module):
    """
    FutureMorph main class. Both predict (inference) and training. 
    given model and path, where to save it
    """
    
    def __init__(self,
        model,
        model_path,
        device="cuda",
        ):
        super().__init__()
        self.device = device
        self.model = model.to(self.device)
        self.model_path = model_path

    
    def predict(self,example_batch,plt_ctrl=False):
        with torch.no_grad():
            self.model.eval()
            test_m,test_f, delta_t,max_delta,seg_m,seg_f,meta_dict= example_batch #15
            load_model_and_plot_losses(self.model,checkpoint_path=self.model_path,plt_loss=False)
            input_img =test_m.unsqueeze(dim=0).to(self.device)
            test_f = test_f.unsqueeze(dim=0).to(self.device)
            seg_m =seg_m.unsqueeze(dim=0).to(self.device)
            seg_f =seg_f.unsqueeze(dim=0).to(self.device)
            delta_t_scaled = delta_t.float()  / max_delta
            
            delta_t_scaled = delta_t_scaled.unsqueeze(dim=0).to(self.device)
            input = input_img 
           
            moved_test, def_test,_ = self.model(input,delta_t_scaled,meta_dict,registration=True)
            if plt_ctrl:
                plot_slices(meta_dict,test_m, test_f,moved_test,def_test,delta_t=delta_t)

            return moved_test, def_test
    
    
    def train(self,optimizer,resume_training,epochs,train_loader,training_iters,loss,writer,add_dice=False):
        print("Saving checkpoint to "+self.model_path)
        self.model.train()
        loss_rec_all = []
        loss_flow_all = []
        loss_total_all = []
        global_step = 0
        image_loss_func = loss['image']
        flow_loss_func = loss['flow']
        ad_sensitive_labels = [17, 53, 4, 43, 5, 44, 3, 42]

        if resume_training:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1  # resume from next epoch
        else:
            start_epoch = 0
        for epoch in range(start_epoch,epochs):  # or more
            loss_rec_sum = 0.0 
            loss_total_sum = 0.0
            loss_flow_sum = 0.0
            for i_iter,batch in enumerate(train_loader):
                #optimizer.zero_grad()
                moving, fixed, meta_data,max_delta,seg_m,seg_f,meta_data_dict = batch
                t_steps_scaled = meta_data.float() / max_delta  # or 7
                #t_steps_scaled = torch.clamp(t_steps_scaled.round(), min=1).long()
                moving = moving.float().to(self.device)
                fixed = fixed.float().to(self.device)
                t_steps_scaled =  t_steps_scaled.to(self.device)
                #delta_t = meta_data.to(device)
                img_input = moving
                input = img_input  #,delta_t,max_delta
                if add_dice:
                    moved,pred_def,flow = self.model(input,t_steps_scaled,meta_data_dict,registration=True)
                else :
                    moved,flow = self.model(input,t_steps_scaled,meta_data_dict,registration=False)
                    

                loss_rec  = image_loss_func(moved,fixed)
                loss_flow = flow_loss_func(y_pred = flow)

                if add_dice:
                   seg_moved =  WrapSeg(seg_m.to(self.device),pred_def,self.device)
                   seg_f = seg_f.squeeze(dim=1)
                   loss_dice =  MultiClassDiceLoss(labels=ad_sensitive_labels).forward(y_true=seg_f.to(self.device),y_pred=seg_moved)
                   loss = 1*loss_rec+ 0.1* loss_flow +1*loss_dice
                else : 
                    loss = 1*loss_rec+ 0.1* loss_flow # +0.5 loss_rec2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if add_dice:
                    print(f"Epoch {epoch},Iter {i_iter},Loss_total: {loss.item():.4f}, Loss_rec: {loss_rec.item():.4f}, Dice: {loss_dice.item():.4f}, Loss_grad:{loss_flow.item():.4f}",flush=True)
                else : 
                    print(f"Epoch {epoch},Iter {i_iter},Loss_total: {loss.item():.4f}, Loss_rec: {loss_rec.item():.4f}, Loss_grad:{loss_flow.item():.4f}",flush=True)
                loss_rec_sum += loss_rec.item()
                loss_total_sum += loss.item()
                loss_flow_sum += loss_flow.item()

                # --- log per iteration (optional, noisy) ---
                if global_step % 10 == 0:
                    writer.add_scalar("Loss/total_iter", loss.item(), global_step)
                    writer.add_scalar("Loss/reconstruction1_iter", loss_rec.item(), global_step)
                    #writer.add_scalar("Loss/reconstruction2_iter", loss_rec_2.item(), global_step)
                    writer.add_scalar("Loss/flow_iter", loss_flow.item(), global_step)

                global_step += 1
                if global_step % 10 == 0:  # log every 200 iterations
                    log_example_images(writer, moving, fixed, moved, i_iter)

            losses_dict = {
                "l_rec": loss_rec_all.append(loss_rec_sum / training_iters),
                "l_flow": loss_flow_all.append(loss_flow_sum / training_iters) ,
                "l_tot": loss_total_all.append(loss_total_sum / training_iters)
                            }


                # --- log per epoch (averaged) ---
            n = len(train_loader)
            writer.add_scalar("Loss/total_epoch", loss_total_sum/n, epoch)
            writer.add_scalar("Loss/reconstruction1_epoch", loss_rec_sum/n, epoch)
            writer.add_scalar("Loss/flow_epoch", loss_flow_sum/n, epoch)
            save_checkpoint(self.model, optimizer, epoch, losses_dict, save_path=self.model_path)
    




            




   
