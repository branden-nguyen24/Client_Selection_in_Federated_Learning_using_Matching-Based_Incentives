class Suited():

    def __init__(self, clientId=None, preferenceList=None,
                 assignedSuitor=None, suitorsList=None):
        """
        Initializes a Suited class
        Args:

        clientId: A unique client ID, identifying the Suited
        preferenceList: A dict of Suitors/servers in the
        format {Suitor:preference} (higher preference = more
        preferred)
        assignedSuitor: The Suitor that the client is assigned
        to for the current round. Needs to be reset after new
        round.
        suitorList: A list of suitor that have requested this
        client in the current round.

        """

        if preferenceList is None:
            self.preferenceList = {}
        else:
            self.preferenceList = preferenceList
        if suitorsList is None:
            self.suitorsList = []
        else:
            self.suitorsList = suitorsList

        self.clientId = clientId
        self.assignedSuitor = assignedSuitor

    def assign(self, suitor):
        """
        Assigns this Suited to a new Suitor's client list
        and assigns itself to the new Suited. If this Suited
        has already been assigned, it will first remove itself
        from the Suitor's assigned client list.

        Args:
        suitor: The new Suitor that the Suited is assigned to.

        """
        if self.assigned is None:
            suitor.addClient(self)
            self.assigned = suitor
        else:
            self.assigned.removeClient(self)
            suitor.addClient(self)
            self.assigned = suitor

    def addSuitor(self, suitor):
        """
        Adds a new Suitor this Suited's suitorsList, if the
        Suitor is not in the Suited's suitorsList.
        Args:
        suitor: The new Suitor to be added to this Suited's
        suitors list.
        Returns:
        True if Suitor is not in the suitorsList and has
        been added.
        False if Suitor already present in suitorsList
        """
        if suitor not in self.suitorsList:
            self.suitorsList.append(suitor)
            return True
        else:
            return False

    def removeSuitor(self, suitor):
        """
        Removes an existing Suitor from this Suited's
        suitorsList, if it exists.
        Args:
        suitor: The Suitor to be removed from this Suited's
        suitors list.
        Returns:
        True if Suited was in the suitorsList and has
        been removed.
        False if Suited is not present in suitorsList.
        """
        if suitor not in self.suitorsList:
            return False
        else:
            self.suitorsList.remove(suitor)
            return True

    def checkPreference(self):
        """
        Checks this Suited's preferenceList against this
        Suited's suitorsList, and returns the preferred
        Suitor.

        Returns:
        The preferred Suitor object, if it exists. If the
        suitorsList preferenceList is empty, or if the suitors
        list doesn't match any of the preferences, returns False

        """
        preferedSuitor = False
        highestPreference = 0
        if not self.suitorsList or not self.preferenceList:
            return False
        for suitor, preference in self.preferenceList:
            if suitor in self.suitorsList and \
                    preference > highestPreference:
                highestPreference = preference
                preferedSuitor = suitor
        return preferedSuitor


class Suitor():

    def __init__(self, serverID=None, preferenceList=None,
                 assignedSuiteds=None, suitedsList=None,
                 capacity=0):
        """
        Initializes a Suitor class
        Args:

        serverId: A unique server ID, identifying the Suited
        preferenceList: A dict of Suiteds/clients in the
        format {Suited:preference} (higher preference = more
        preferred)
        assignedSuiteds: A list of Suiteds that are assigned to
        the Suitor for the current round. Needs to be reset
        after new round.
        suitedsList: A list of suiteds that have requested this
        Suitor in the current round.
        capacity: Maximum number of Suiteds for this Suitor.

        """
        if preferenceList is None:
            self.preferenceList = {}
        else:
            self.preferenceList = preferenceList
        if suitedsList is None:
            self.suitedsList = []
        else:
            self.suitedsList = suitedsList
        if assignedSuiteds is None:
            self.assignedSuiteds = []
        else:
            self.assignedSuiteds = assignedSuiteds

        self.serverID = serverID
        self.assignedSuiteds = assignedSuiteds
        self.capacity = capacity

    def addSuited(self, suited):
        """
        Adds a new Suited to this Suitor's suitedsList, if the
        Suited is not in this Suitor's suitedsList.
        Args:
        suited: The new Suited to be added to this Suitor's
        suitedslist.
        Returns:
        True if Suited is not in the suitedsList and has
        been added.
        False if Suited already present in suitedsList
        """
        if suited not in self.suitedsList:
            self.suitedsList.append(suited)
            return True
        else:
            return False

    def removeSuited(self, suited):
        """
        Removes an existing Suited from this Suitor's
        suitedsList, if it exists.
        Args:
        suited: The Suitor to be removed from this Suitor's
        suitedsList.
        Returns:
        True if Suited was in the suitedsList and has
        been removed.
        False if Suited is not present in suitedsList.
        """
        if suited in self.suitedsList:
            self.suitedsList.remove(suited)
            return True
        else:
            return False

    def notFull(self):
        """
        Checks if the Suitor is at capacity

        Returns:
        True if the Suitor is not at capacity for
        assigned suiteds
        False, if it is not at capacity
        """
        if (len(self.suitedsList) < self.capacity):
            return True
        else:
            return False

    def checkPreference(self):
        """
        Checks this Suited's preferenceList against this
        Suited's suitorsList, and returns the preferred
        Suitor.

        Returns:
        The preferred Suitor object, if it exists. If the
        suitorsList preferenceList is empty, or if the suitors
        list doesn't match any of the preferences, returns False

        """
        preferedSuiteds = {}

        if not self.suitedsList or not self.preferenceList:
            return False
        for suited, preference in self.preferenceList:
            if suited in self.suitedsList:
                preferedSuiteds.update({suited, preference})

            sortedPreferedSuiteds = sorted(preferedSuiteds.items(),
                                           key=lambda x: x[1])

        return sortedPreferedSuiteds[0:self.capacity]


def match(suitorList: list[Suitor], suitedList: list[Suited]):
    continueMatching = True

    while continueMatching:
        for server in suitorList:
            if server.notFull():
                selected_client = max(server.preferenceList,
                                      key=server.preferenceList.get)
                server.preferenceList.pop(selected_client)
                selected_client.addSuitor(server)
        for client in suitedList:
            clientPref = client.checkPreference
            client.assign(clientPref)
        
        continueMatching = False
        for server in suitorList:
            if server.notFull():
                continueMatching = True

        
