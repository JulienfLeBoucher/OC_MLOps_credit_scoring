from typing import List
import pandas as pd
from seaborn import kdeplot, scatterplot
import matplotlib.pyplot as plt
from numpy import isfinite


class Group:
    """ Group data structure that contains :
    - a name.
    - a description.
    - a grouper which is a list of features to group by.
    
    Facilitates :
    - data selection of a group of people similar to a certain customer.
    - plotting those data. """
    
    def __init__(
        self, 
        name: str,
        description: str,
        grouper: List[str],
    ):
        """grouper is the list of features to be used when grouping by."""
        # Attributes
        self.name = name 
        self.description = description 
        self.grouper = grouper 
        
    def get_group_data(self, data: pd.DataFrame, customer_id: int):
        """ get data of the group the customer is in"""
        if self.grouper is None:
            return data
        else:
            for _, group in data.groupby(self.grouper).__iter__():
                if customer_id in group.index:
                    return group
                
    def plot_feature_kde_with_client_value(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        customer_id: int,
        feature_name: str,
    ):
        """ Return the figure containing the kdeplots of a particular 
        feature observed in the Group, colored by customers class
        (payment risk or not). 
        Also pinpoint the customer value if not null."""
        # Replace target per class description and make it categorical
        target = (
            target
            .astype("category")
            .replace(0, 'without repayment risk')
            .replace(1, 'with repayment risk')
        )
        # get group relative data 
        group_data = self.get_group_data(features_df, customer_id)
        if len(group_data) <= 3:
            # There is no fig when it can not be meaningful
            # because there are to few people
            return None
        else:
            # Join the relevant feature of the group and the target,
            # and make sure to keep only group data.
            group_feature = group_data.loc[:, feature_name]
            group_feature = (
                pd.concat([group_feature, target], axis=1)
                .loc[group_data.index, :]
            )
            # Rename the target columns for legend display.
            group_feature = (
                group_feature
                .rename(
                    columns={'TARGET': 'Customer type'},
                )
            )
            # Get customer feature value.
            customer_val = group_feature.loc[customer_id, feature_name]
            # plot 
            fig, ax = plt.subplots(figsize=(8,4))
            kdeplot(
                data=group_feature,
                x=feature_name,
                hue='Customer type',
                ax=ax,
                legend=True,
            )
            # Custom 
            title = (
                f'Distributions per customer type of {feature_name}\n'
                f'considering {self.description.lower()}\n'
            )
            ax.get_yaxis().set_visible(False)
            _, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()
            # Draw an arrow pinpointing the customer value, if there
            # is one.
            if isfinite(customer_val):
                ax.annotate(
                    'Customer value',
                    xy=(customer_val, 0),
                    xytext=(customer_val, 0.15*y_max),
                    arrowprops=dict(facecolor='gray', shrink=0.05),
                    ha='center',
                )
            else:
                ax.annotate(
                    'Unknown Customer value',
                    xy=((x_min + x_max)/2, 0.05*y_max),
                    ha='center',
                )
            plt.title(title)
            return fig
        
    def plot_scatter_with_client_value(
        self,
        features_df,
        target,
        customer_id,
        feature_name_x,
        feature_name_y,
    ):
        """ Return a scatter plot colored by customers class
        (payment risk or not). Also pinpoint the customer value if not null."""
        # Replace target per class description and make it categorical
        target = (
            target
            .astype("category")
            .replace(0, 'without repayment risk')
            .replace(1, 'with repayment risk')
        )
        # get group relative data 
        group_data = self.get_group_data(features_df, customer_id)
        if len(group_data) <= 3:
            # There is no fig when it can not be meaningful
            # because there are to few people
            return None
        else:
            # Join the relevant feature of the group and the target,
            # and make sure to keep only group data.
            feature_names = [feature_name_x, feature_name_y]
            group_features = group_data.loc[:, feature_names]
            group_features = (
                pd.concat([group_features, target], axis=1)
                .loc[group_data.index, :]
            )
            # Rename the target columns for legend display.
            group_features = (
                group_features
                .rename(
                    columns={'TARGET': 'Customer type'},
                )
            )
            # Get customer feature value.
            customer_val_x = group_features.loc[customer_id, feature_name_x]
            customer_val_y = group_features.loc[customer_id, feature_name_y]
            # plot 
            fig, ax = plt.subplots(figsize=(6,5))
            scatterplot(
                data=group_features,
                x=feature_name_x,
                y=feature_name_y,
                hue='Customer type',
                ax=ax,
                legend=True,
            )
            # Custom 
            title = (
                f'{feature_name_x} vs {feature_name_y}\n'
                f'considering {self.description.lower()}\n'
            )
            # ax.get_yaxis().set_visible(False)
            y_min, y_max = ax.get_ylim()
            span_y = y_max - y_min 
            x_min, x_max = ax.get_xlim()
            span_x = x_max - x_min
            # Draw a red circle to indicate the customer value, if there
            # is one.
            if (isfinite(customer_val_x) and isfinite(customer_val_y)):
                plt.plot(customer_val_x, customer_val_y,'mo')
                ax.annotate(
                    'Customer value',
                    xy=(customer_val_x, customer_val_y),
                    xytext=(
                        customer_val_x + 0.07 * span_x,
                        customer_val_y + 0.07 * span_y
                    ),
                    color='m',
                    arrowprops=dict(facecolor='magenta', shrink=0.05),
                    ha='left',
                )
            else:
                ax.annotate(
                    'Unknown Customer value',
                    color='m',
                    xy=((x_min + x_max)/2, y_min + 0.05* span_y),
                    ha='center',
                )
            plt.title(title)
            return fig